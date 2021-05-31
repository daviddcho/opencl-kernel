DeviceInfo: DeviceInfo.c
	gcc -o DeviceInfo DeviceInfo.c -framework OpenCL
vadd: vadd.c wtime.c device_info.c
	gcc -o vadd -O3 -lm vadd.c wtime.c device_info.c -framework OpenCL
chain_vadd: chain_vadd.c wtime.c device_info.c
	gcc -o chain_vadd -O3 -lm chain_vadd.c wtime.c device_info.c -framework OpenCL
matmul: matmul.c wtime.c device_info.c 
	gcc -o matmul -O3 -lm matmul.c wtime.c device_info.c -framework OpenCL
