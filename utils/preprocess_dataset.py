import argparse
import glob
import json
import os
import pdb
import pprint
import h5py
import tqdm

# Usage Locally:
# python scripts/preprocess_dataset.py --dataset_lists object_nav_v0.10 --dataset_path experiment_output/dataset

# Usage Beaker:
# python scripts/preprocess_dataset.py --dataset_lists simple_explore_house_v0.13 --dataset_path /data/datasets
# python scripts/preprocess_dataset.py --dataset_lists ObjectNavType --dataset_path /net/nfs.cirrascale/prior/datasets/fpin_datasets/all_variations

# Usage work-station:
# python scripts/preprocess_dataset.py --dataset_lists ALL --dataset_path /net/nfs.cirrascale/prior/datasets/vida_datasets/multi_task_v2/ --override
# python scripts/preprocess_dataset.py --dataset_lists ALL --dataset_path /net/nfs.cirrascale/prior/datasets/vida_datasets/multi_task_allfetch_2023_10_05_CLOSED_OBJAVERSE/ --override

# /net/nfs2.prior/datasets/vida_datasets/multi_task_2023_09_22_CLOSED_OBJAVERSE/


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess")
    parser.add_argument("--dataset_lists", nargs="+", default=[])
    parser.add_argument("--dataset_path", type=str, default="")
    parser.add_argument("--override", action="store_true")
    parser.add_argument("--limit_episodes", type=int, default=None)
    args = parser.parse_args()
    if "ALL" in args.dataset_lists:
        assert len(args.dataset_lists) == 1
        args.dataset_lists = [
            dataset_name.split("/")[-1]
            for dataset_name in glob.glob(os.path.join(args.dataset_path, "*"))
        ]
    return args


def preprocess_dataset(dataset_path):
    all_hdf_files = glob.glob(os.path.join(dataset_path, "*/hdf5_sensors.hdf5"))
    result_dict = {}
    broken_ones = []
    for hdf_file in tqdm.tqdm(all_hdf_files):
        if args.limit_episodes is not None and len(result_dict) >= args.limit_episodes:
            break
        try:
            with h5py.File(hdf_file, mode="r") as hdf5_file:
                house_id = hdf_file.split("/")[-2]
                episode_names = [k for k in hdf5_file.keys()]
                # For each trajectory, check that both videos exist
                for episode_name in episode_names:
                    nav_adr = os.path.join(
                        dataset_path, house_id, f"raw_navigation_camera__{episode_name}.mp4"
                    )
                    manip_adr = os.path.join(
                        dataset_path, house_id, f"raw_manipulation_camera__{episode_name}.mp4"
                    )
                    if os.path.exists(nav_adr) and os.path.exists(manip_adr):
                        # If they do, add to result_dict
                        result_dict.setdefault(house_id, [])
                        result_dict[house_id].append(episode_name)
                    else:
                        broken_ones.append((house_id, episode_name))
        except Exception as e:
            print(f"Failed to process file {hdf_file} due to error: {e}")
            continue
    print("failed for total of", len(broken_ones))
    return result_dict


def main(args):
    final_stat = {}
    for dataset in args.dataset_lists:
        final_stat[dataset] = {}
        print(f"Preprocessing dataset {dataset}")
        for split in ["val", "train", "test"]:
            print("Split", split)

            dataset_direct_path = os.path.join(args.dataset_path, dataset)
            json_path = os.path.join(dataset_direct_path, f"house_id_to_sub_house_id_{split}.json")
            split_path = os.path.join(dataset_direct_path, split)

            if not args.override and os.path.exists(json_path):
                print(f"Path {json_path} already exists")
                continue

            if not os.path.exists(split_path):
                print(f"Path {split_path} does not exist")
                continue
            dataset_dict = preprocess_dataset(split_path)

            final_stat[dataset][split] = dict(
                num_houses=len(dataset_dict),
                num_episodes=sum([len(dataset_dict[k]) for k in dataset_dict]),
            )
            print(
                "FINAL STAT",
            )
            pprint.pprint(final_stat)

            with open(json_path, "w") as f:
                json.dump(dataset_dict, f)


if __name__ == "__main__":
    args = parse_args()
    main(args)
