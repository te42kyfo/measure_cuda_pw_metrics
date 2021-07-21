

measureMetricPW.so: measureMetricPW.cpp
	g++ measureMetricPW.cpp -fPIC -shared -o measureMetricPW.so -L$(CUDA_HOME)/lib64 -lcuda -lcupti -lnvperf_host -lnvperf_target -I$(CUDA_HOME)/include -I$(CUDA_HOME)/extras/CUPTI/include -L$(CUDA_HOME)/extras/CUPTI/lib64 $(shell python3-config --includes) -Wl,-rpath=$(CUDA_HOME)/extras/CUPTI/lib64

clean:
	@rm measureMetricPW.so
