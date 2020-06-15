#!/bin/bash

schedRun() {
   #lockfile -r 0 /tmp/verity.lock || exit 1
   if [ ! -f verity.lock ]; then
       echo "fresh run!!!"
       echo "running" > verity.lock
       python VerityIntegratedScoring.py > "logs/$(date +"%Y_%m_%d_%I_%M_%p").txt"
       rm verity.lock
   else
       echo "Still running"
   fi
	   
}

while [ 1 ]
do
    schedRun
    sleep 7200
done