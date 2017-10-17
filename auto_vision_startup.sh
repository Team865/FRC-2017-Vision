#!/bin/bash

check_process(){
	if ["$1" = ""];
	then 
		return 0
	fi

	PROCESS_NUM=$(ps -ef | grep "$1" | grep -v "grep" | wc -l)
	
	if [$PROCESS_NUM -eq 1];
	then 
		return 1
	else 
		return 0
	fi 
}

while true; do 
	
	CHECK_RET=check_process "python target_testing.py 10.8.65.91"
	if [ $CHECK_RET -eq 0];
	then 
		python target_testing.py 10.8.65.91
	fi
	sleep 3
done
