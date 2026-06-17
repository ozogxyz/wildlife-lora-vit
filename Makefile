# GCP GPU VM. Edit over TRAMP: /ssh:vm.me-west1-c.datadriven-499611:/home/motorbreath/wildlife-lora-vit/
# Run with `make run` (local M-x compile) — ssh over the master socket, ~0.6s, not TRAMP's 3.4s async.
.PHONY: start stop addr ssh run
VM   := vm
ZONE := me-west1-c
HOST := vm.me-west1-c.datadriven-499611
DIR  := wildlife-lora-vit
ARGS := --frac 0.01 --epochs 2

start: ; gcloud compute instances start $(VM) --zone=$(ZONE) && gcloud compute config-ssh
stop:  ; gcloud compute instances stop  $(VM) --zone=$(ZONE)
addr:  ; gcloud compute config-ssh
ssh:   ; gcloud compute ssh $(VM) --zone=$(ZONE)
run:   ; ssh $(HOST) 'cd $(DIR) && python3 train.py $(ARGS)'
