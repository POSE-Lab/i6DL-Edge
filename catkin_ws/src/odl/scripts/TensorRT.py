import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
import numpy as np
from profiler import Profiler
from PIL import Image

class TRT_engine:
    def __init__(self,engine_path,input_tensor,profiler : Profiler) -> None:
        
        self.profiler = profiler
        self.cfx = cuda.Device(0).make_context()
        self.runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING)) 
        with open(engine_path, 'rb') as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        input_shape = self.context.get_tensor_shape(input_tensor)
        input_nbytes = trt.volume(input_shape) *np.dtype(np.float32).itemsize

        self.input_gpu = cuda.mem_alloc(input_nbytes)

        self.stream = cuda.Stream()

        #Allocate output buffer
        self.cpu_outputs=[]
        self.gpu_outputs=[]
        for i in range(1,5):
            self.cpu_outputs.append(cuda.pagelocked_empty(tuple(self.context.get_binding_shape(i)), dtype=np.float32))
            self.gpu_outputs.append(cuda.mem_alloc(self.cpu_outputs[i-1].nbytes))

    def predict(self,image):

        self.profiler.addTrackItem("trt_prediction_time")
        self.profiler.start("trt_prediction_time")
        self.cfx.push()
        
        cuda.memcpy_htod_async(self.input_gpu,image,self.stream)
        #Copy inouts
        self.context.execute_async_v2(bindings=[int(self.input_gpu)] + [int(outp) for outp in self.gpu_outputs] , 
            stream_handle=self.stream.handle)

        for i in range(4):
            cuda.memcpy_dtoh_async(self.cpu_outputs[i], self.gpu_outputs[i], self.stream)

        self.stream.synchronize()


        eval_time_elapsed = self.profiler.stop("trt_prediction_time")

        predictions={'pred_frag_conf':self.cpu_outputs[2],
                    'pred_frag_loc':self.cpu_outputs[1],
                    'pred_obj_conf':self.cpu_outputs[0],
                    'pred_obj_label':self.cpu_outputs[3]}
        self.cfx.pop()
        return predictions,eval_time_elapsed
    
    def visualize(self):
        
        np_image = np.array(self.cpu_outputs[3])
        np_image[np_image !=0]=255
        aa = np.reshape(np_image.astype("uint8"),(180,320))
        image = Image.fromarray(aa)
        image.show()
