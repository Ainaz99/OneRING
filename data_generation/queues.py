import abc
import math
import queue
import uuid
from typing import Dict, Union

import boto3
from botocore.exceptions import ClientError
from torch import multiprocessing as mp
from torch.distributions.utils import lazy_property

from allenact.utils.system import get_logger

DEFAULT_SQS_MESSAGE_VISIBILITY_TIMEOUT = 60 * 20  # 20 mins


class Message:
    body: str
    attributes: Dict[str, Dict[str, str]]
    message_id: str
    receipt_handle: str

    def delete(self):
        raise NotImplementedError


class QueueMessage(Message):
    def __init__(self, body: str, attributes: Dict[str, Dict[str, str]]):
        self.body = body
        self.attributes = attributes
        self.message_id = str(uuid.uuid4())
        self.receipt_handle = str(uuid.uuid4())

    def __repr__(self):
        return f"Message({self.message_id}, {self.body}, {self.attributes})"

    def delete(self):
        pass


class LocalOrRemoteQueue(abc.ABC):
    @abc.abstractmethod
    def get(self, timeout: float = None) -> Message:
        raise NotImplementedError

    @abc.abstractmethod
    def mark_complete(self, message: Message):
        raise NotImplementedError

    @lazy_property
    def message_timeout(self) -> Union[float, int]:
        raise NotImplementedError

    def refresh_message(self, message: Message, new_timeout: int):
        raise NotImplementedError


class FromToQueue(LocalOrRemoteQueue):
    def __init__(self, from_queue: mp.Queue, to_queue: mp.Queue):
        self.from_queue = from_queue
        self.to_queue = to_queue

    def get(self, timeout: float = None) -> Message:
        item = self.from_queue.get(timeout=timeout)
        message = QueueMessage(body=item, attributes={})
        return message

    def mark_complete(self, message: Message):
        return self.to_queue.put(message.body)

    @lazy_property
    def message_timeout(self) -> Union[float, int]:
        return float("inf")

    def refresh_message(self, message: Message, new_timeout: int):
        pass


class SQSQueueWrapper(LocalOrRemoteQueue):
    def __init__(self, queue_name: str):
        self.queue_name = queue_name

    @lazy_property
    def sqs_resource(self):
        return boto3.resource("sqs")

    @lazy_property
    def sqs_client(self):
        return boto3.client("sqs")

    @lazy_property
    def message_timeout(self) -> Union[float, int]:
        # return int(self.boto3_sqs_queue.get_attributes(Attributes=["VisibilityTimeout"]))
        return DEFAULT_SQS_MESSAGE_VISIBILITY_TIMEOUT

    @lazy_property
    def boto3_sqs_queue(self):
        return self.get_queue(self.queue_name)

    def get_queue(self, name: str):
        """Gets an SQS queue by name.

        :param name: The name that was used to create the queue.
        :return: A Queue object.
        """
        try:
            queue = self.sqs_resource.get_queue_by_name(QueueName=name)
            print(f"Got queue '{name}' with URL={queue.url}")
        except ClientError:
            print(f"Couldn't get queue named {name}.")
            raise
        else:
            return queue

    def get(self, timeout: float = 20.0) -> Message:
        """Receive a message in a single request from an SQS queue."""
        timeout = int(math.ceil(max(min(timeout, 20), 0)))
        try:
            messages = self.boto3_sqs_queue.receive_messages(
                MessageAttributeNames=["All"],
                MaxNumberOfMessages=1,
                WaitTimeSeconds=timeout,
            )
            if len(messages) == 0:
                raise queue.Empty
            message = messages[0]
        except ClientError:
            get_logger().error(f"Couldn't receive messages from queue: {self.boto3_sqs_queue}")
            raise

        return message

    def mark_complete(self, message: Message):
        try:
            message.delete()
        except ClientError:
            get_logger().error(f"Couldn't delete message: {message.message_id}")
            raise

    def refresh_message(self, message: Message, new_timeout: int):
        try:
            self.sqs_client.change_message_visibility(
                QueueUrl=self.boto3_sqs_queue.url,
                ReceiptHandle=message.receipt_handle,
                VisibilityTimeout=new_timeout,
            )
        except ClientError:
            get_logger().error(
                f"Couldn't change visibility timeout for message: {message.message_id}"
            )
            raise


class SQSQueueWrapperImmediateComplete(SQSQueueWrapper):
    def mark_complete(self, message: Message):
        raise NotImplementedError

    def get(self, timeout: float = 20.0) -> str:
        message = super().get(timeout=timeout)
        super(SQSQueueWrapperImmediateComplete, self).mark_complete(message)
        return message.body.strip()
