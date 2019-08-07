# Linux-Environment
Linux Environment, GPU CUDA

Error & Solutions
tensorflow.python.framework.errors_impl.NotFoundError: /search/odin/yangjian/software/anaconda3/lib/python3.6/site-packages/horovod/tensorflow/mpi_lib.cpython-36m-x86_64-linux-gnu.so: undefined symbol: _ZTVN10tensorflow14kernel_factory17OpKernelRegistrar18PtrOpKernelFactoryE <br>
==> checkout https://github.com/horovod/horovod/issues/656  <br>
```
$ pip uninstall -y horovod
$ conda install gcc_linux-64 gxx_linux-64
$ [flags] pip install --no-cache-dir horovod
```
