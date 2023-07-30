
import wandb
import argparse
import os
import glob

if __name__ == '__main__':

    # Visualize a single sweep
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--team_name',
        type=str,
        default='wong-n',
        help='The wandb team name.'
    )
    parser.add_argument(
        '--project_name',
        type=str,
        default='mnll-onehots',
        help='The wandb project name.'
    )
    parser.add_argument(
        '--group_name',
        required=True,
        type=str,
        help='The wandb group name, which matches the name of folder in /logs.'
    )
    args = parser.parse_args()

    LOGS_PATH = 'logs'
    group_path = os.path.join(LOGS_PATH, args.group_name)
    assert os.path.exists(group_path), f"Group name {args.group_name} does not have a log set at /logs."

    run_paths = glob.glob(f'{group_path}/*/*')
    extract_from_path = lambda path, index : path.split('/')[index]
    extract_id = lambda path : extract_from_path(path, -1)
    extract_run_name = lambda path : extract_from_path(path, -2)
    run_ids = list(map(extract_id, run_paths))
    run_names = list(map(extract_run_name, run_paths))

    for path, run_name, run_id in zip(run_paths, run_names, run_ids):
        api = wandb.Api()
        run = api.run(f'{args.team_name}/{args.project_name}/{run_id}')
        history = run.history()
        history.to_csv(os.path.join(path, 'history.csv'))
