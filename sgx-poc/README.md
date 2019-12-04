# Proof of concept APA Attack against SGX

This directory contains all the required code to run the proof-of-concept APA attack against real SGX hardware.
In particular, we provide three components:

- *kernel-for-nukemod*: this repository contains a custom Linux kernel which is minimally modified to allow hooking the page fault handler, which we do in our attack.
- *nukemod*: this repository contains a Linux kernel module which contains the logic of our APA attack, including halting and rescheduling SGX threads.
- *sgx_scheduling*: this repository contains our C-based proof-of-concept SGX code which mimics the main operations in the PyTorch code.
