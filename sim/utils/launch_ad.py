import subprocess
import time
import os
import shutil


def launch(shell_path, cuda_id, output, extra_env=None):
    os.makedirs(output, exist_ok=True)
    print(os.path.join(output, 'output.txt'))
    print(shell_path, cuda_id, output)
    env = os.environ.copy()
    if extra_env:
        env.update({key: str(value) for key, value in extra_env.items() if value is not None})
    shell = shutil.which("zsh") or shutil.which("bash") or "bash"
    with open(os.path.join(output, 'output.txt'), 'w') as f:
        process = subprocess.Popen(
            [shell, shell_path, cuda_id, output], stdout=f, stderr=f, env=env
        )
    return process


def check_alive(process, tolerant=100):
    i = 0
    while i < tolerant:
        return_code = process.poll()
        if return_code is not None:
            print(f"The AD algorithm completed with return code {return_code}.")
            process.kill()
            return
        elif i % 5 == 0:
            print(f"The AD algorithm is still running, remaining tolerant {tolerant - i}.")
        time.sleep(1)
        i += 1
    process.kill()
    print("The AD algorithm process is killed.")
