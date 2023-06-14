import argparse
import json
import copy
import os
import subprocess


def convert_dict_to_command_line_args(dict):
    return(' '.join(['--'+key + ' ' + str(value) for key,value in dict.items()]))


def gen_one_model_submission_script(model, singularity_base_command, slurm_params):    
    
    commands = []
    commands.append("#!/bin/bash -e\n")
    
    commands.append("\n#SBATCH -N 1\n")                         
    commands.append(f"#SBATCH -p {slurm_params['partition']}\n")    
    commands.append(f"#SBATCH -t {slurm_params['time_alloc_hrs_str']}\n")
    commands.append(f"#SBATCH --mem={slurm_params['mem_alloc_gb']}G\n")
    commands.append("#SBATCH --gres=gpu:1\n")
    commands.append(f"#SBATCH --constraint={slurm_params['gpu_constraint']}\n")
    commands.append(f"#SBATCH --exclude={slurm_params['exclusion_list']}\n")
    commands.append(f"#SBATCH --ntasks={slurm_params['n_tasks']}\n")
    commands.append(f"#SBATCH --cpus-per-task={slurm_params['cpus_per_task']}\n")
    commands.append(f"#SBATCH --output={slurm_params['slurm_folder']}/%j.out\n")    
    commands.append(f"\nmkdir -p {slurm_params['slurm_folder']}\n")    
    commands.append(f"\nmodule load {slurm_params['singularity_version']}\n")        
    commands.append('mkdir -p ~/.cache/$SLURM_JOB_ID\n')
    
    # append the actual command    
    command = singularity_base_command + ' '+ slurm_params['script'] + ' ' + convert_dict_to_command_line_args(model)
    commands.append(f"\n{command}\n")

    # need to make sure that the directory exists
    os.system(f"mkdir -p SLURM/{model['wandb_group']}")
    model_file = os.path.join('SLURM',model['wandb_group'],model['wandb_id']+'.sh')

    # write out the commands
    with open(model_file, 'w') as f: f.writelines(commands)
    print('Wrote out commands to '+model_file)

    return(model_file)

def gen_all_models_submission_script(wandb_group, model_file_paths):
    text = ['#!/bin/bash -e\n']
    for model_file_path in  model_file_paths:
        text.append('sbatch '+os.path.basename(model_file_path)+'\n')
        
    submit_sh_path = os.path.join('SLURM', wandb_group, 'submit.sh') 
    
    # make sure that the path exists
    submit_sh_dir = os.path.dirname(submit_sh_path)
    if not os.path.exists(submit_sh_dir):
        os.makedirs(submit_sh_dir)
    
    with open(submit_sh_path, 'w') as f:
        f.writelines(text)
    
    subprocess.call(f'chmod u+x {submit_sh_path}', shell = True)


if __name__ == "__main__":
    # Training Arguments
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--model_spec',
        type=str,
        required=True,
        help='path of the JSON with the set of models that need to be run'
    )

    parser.add_argument(
        '--sandbox',
        action='store_true',
        help="Generate OM submission scripts but don't actually submit them to the SLURM scheduler"
    )

    args = parser.parse_args()
    with open(args.model_spec, 'r') as f:
        model_spec = json.load(f)

    # load the defaults
    defaults = model_spec['defaults'] 
    model_listing = model_spec['models'] 
    models_to_run = [] 

    # add the individual models
    for model in model_listing:
        run_model_spec = copy.copy(defaults)
        for key,value in model.items():
            run_model_spec[key] = value        
        models_to_run.append(run_model_spec)


    singularity_base_command = "singularity exec -B /om2:/om2 --env PYTHONPATH=/usr/local/lib/python3.8/dist-packages --nv /home/smeylan/om2/vagrant/menll_pytorch.sbx python3"

    model_file_paths = [gen_one_model_submission_script(model, singularity_base_command, model_spec['slurm']) for model in models_to_run]
    
    gen_all_models_submission_script(defaults['wandb_group'], model_file_paths)

