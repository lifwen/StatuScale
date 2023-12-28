import sys
import os
import subprocess

if __name__ == "__main__":
    args = sys.argv[1:]
    os.system("echo 30000 > $(find $(find /sys/fs/cgroup/cpu,cpuacct/kubepods.slice/ -name *" + args[1][:8] + "*) -maxdepth 1 -name cpu.cfs_quota_us)")
    cmd="find $(find $(find /sys/fs/cgroup/cpu,cpuacct/kubepods.slice/ -name *" + args[1][:8] + "*) -maxdepth 1 -name docker*)  -maxdepth 1 -name cpu.cfs_quota_us"
    result = subprocess.check_output(cmd, shell=True)
    output = result.strip().split('\n')
    for i in range(0, len(output)):
        os.system("echo " + args[0] + " > " + output[i])