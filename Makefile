# GCP GPU VM lifecycle. Train: `make ssh` in, then `python3 main.py <args>`.
.PHONY: start stop addr ssh
VM   := vm2
ZONE := europe-west6-b

start: ; gcloud compute instances start $(VM) --zone=$(ZONE) && gcloud compute config-ssh
stop:  ; gcloud compute instances stop  $(VM) --zone=$(ZONE)
addr:  ; gcloud compute config-ssh
ssh:   ; gcloud compute ssh $(VM) --zone=$(ZONE)
