# GCP GPU VM. Edit over TRAMP: /ssh:vm.me-west1-c.datadriven-499611:/home/motorbreath/datadriven/
.PHONY: start stop addr ssh
VM   := vm
ZONE := me-west1-c

start: ; gcloud compute instances start $(VM) --zone=$(ZONE) && gcloud compute config-ssh
stop:  ; gcloud compute instances stop  $(VM) --zone=$(ZONE)
addr:  ; gcloud compute config-ssh
ssh:   ; gcloud compute ssh $(VM) --zone=$(ZONE)
