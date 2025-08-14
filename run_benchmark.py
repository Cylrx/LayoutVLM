import os 
import subprocess
import sys
import argparse
from concurrent.futures import ThreadPoolExecutor

abs_path = lambda path: os.path.abspath(path)

def layoutvlm_cmd(task, task_dir, save_path, asset_dir, openai_api_key): 
    save_dir = os.path.join(save_path, task)
    script_path = abs_path("./main.py")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return f"python {script_path} --scene_json_file {task_dir} --openai_api_key {openai_api_key} --asset_dir {asset_dir} --save_dir {save_dir}"

def render_cmd(task, task_dir, save_path, asset_dir):
    save_dir = os.path.join(save_path, task)
    assert os.path.exists(save_dir)
    script_path = abs_path("./scripts/render_layout_in_blender.py")
    return f"python {script_path} --scene_json_file {task_dir} --layout_json_file {save_dir}/layout.json --asset_dir {asset_dir} --save_dir {save_dir} --high_res"


def get_tasks(tasks_dir):
    tasks = {}
    tasks_dir = abs_path(tasks_dir)

    for folder in os.listdir(tasks_dir):
        for file in os.listdir(os.path.join(tasks_dir, folder)):
            if file.endswith(".json"):
                task_name = file.split(".")[0]
                tasks[task_name] = os.path.join(tasks_dir, folder, file)
    
    return tasks

def execute_task(task, task_dir, save_path, asset_dir, openai_api_key):
    """Execute cmd1 and cmd2 for a single task."""
    cmd1 = layoutvlm_cmd(task, task_dir, save_path, asset_dir, openai_api_key)
    cmd2 = render_cmd(task, task_dir, save_path, asset_dir)

    print(f"[{task}] Running cmd1...")
    while True: 
        result1 = subprocess.run(cmd1, shell=True, capture_output=True, text=True)
        if result1.returncode == 0:
            break
        print(f"\033[91m[{task}] Failed. Likely due to errors in LLM generated sandbox code\033[0m")
        print(f"[{task}] Retrying cmd1 ...")

    print(f"[{task}] Running cmd2...")
    result2 = subprocess.run(cmd2, shell=True, capture_output=True, text=True)
    if result2.returncode != 0:
        print(f"ERROR in cmd2 for task {task}:")
        print(f"Return code: {result2.returncode}")
        print(f"STDERR:\n{result2.stderr}")
        print(f"STDOUT:\n{result2.stdout}")
        return False
    
    print(f"[{task}] Completed.")
    return True

def main(): 
    parser = argparse.ArgumentParser(description="Run benchmark with concurrent task execution")
    parser.add_argument("-n", "--workers", type=int, default=4, help="Number of concurrent workers (default: 4)")
    args = parser.parse_args()

    tasks = get_tasks("./benchmark_tasks")
    save_path = os.path.abspath("./benchmark_results")
    asset_dir = os.path.abspath("./assets/test_asset_dir")
    openai_api_key = input("Enter your OpenAI API key: ")

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(execute_task, task, task_dir, save_path, asset_dir, openai_api_key)
            for task, task_dir in tasks.items()
        ]
        
        failed_tasks = []
        for i, future in enumerate(futures):
            task_name = list(tasks.keys())[i]
            try:
                success = future.result()
                if not success:
                    failed_tasks.append(task_name)
            except Exception as e:
                print(f"ERROR: Task {task_name} failed with exception: {e}")
                failed_tasks.append(task_name)
        
        if failed_tasks:
            print(f"\nFailed tasks: {failed_tasks}")
            sys.exit(1)

if __name__ == "__main__":
    main()