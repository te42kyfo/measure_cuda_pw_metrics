

measureMetricPW.so: measureMetricPW.cpp
	g++ measureMetricPW.cpp -fPIC -shared -o measureMetricPW.so -L$(CUDA_HOME)/lib64 -lcuda -lcupti -lnvperf_host -lnvperf_target -I$(CUDA_HOME)/include $(shell python3-config --includes)

clean:
	@rm measureMetricPW.so
