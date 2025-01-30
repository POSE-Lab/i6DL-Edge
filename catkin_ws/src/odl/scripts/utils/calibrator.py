import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
from PIL import Image
import ctypes
import tensorrt as trt
from random import shuffle
import glob
import argparse,os

parser = argparse.ArgumentParser()
parser.add_argument("--calib_dataset_loc",required=True,type=str)
parser.add_argument("--saveCache",required=True,type=str)
parser.add_argument("--onnx",required=True,type=str)
parser.add_argument("--img_size", required=True, nargs='+', type=int)
parser.add_argument("--num_samples",default=300,type=int)
parser.add_argument("--batch_size",default=64,type=int)
args = parser.parse_args()

CHANNEL = args.img_size[2]
HEIGHT = args.img_size[0]
WIDTH = args.img_size[1]


TRT_LOGGER = trt.Logger()

class PythonEntropyCalibrator(trt.IInt8EntropyCalibrator2):
  def __init__(self,stream,cache_file):
    trt.IInt8EntropyCalibrator2.__init__(self)     
    self.cache_file = cache_file
    self.stream = stream
    self.d_input = cuda.mem_alloc(self.stream.calibration_data.nbytes)
    stream.reset()

  def get_batch_size(self):
    return self.stream.batch_size

  def get_batch(self,names):
    batch = self.stream.next_batch()
    if not batch.size:   
       return None
      
    cuda.memcpy_htod(self.d_input, batch)
    return [int(self.d_input)]


  def read_calibration_cache(self):
    return None

  def write_calibration_cache(self, cache):
    with open(self.cache_file, "wb") as f:
        f.write(cache)

class ImageBatchStream():
  def __init__(self, batch_size, calibration_files):
    self.batch_size = batch_size
    self.max_batches = (len(calibration_files) // batch_size) + \
                       (1 if (len(calibration_files) % batch_size) \
                        else 0)
    self.files = calibration_files
    self.calibration_data = np.zeros((batch_size, HEIGHT, WIDTH, CHANNEL), \
                                     dtype=np.float32)
    self.batch = 0


  @staticmethod
  def read_image_chw(path):
    im = np.array(Image.open(path)).astype(np.float32)
    return im
         
  def reset(self):
    self.batch = 0
     
  def next_batch(self):
    if self.batch < self.max_batches:
      imgs = []
      files_for_batch = self.files[self.batch_size * self.batch : \
                        self.batch_size * (self.batch + 1)]
      for f in files_for_batch:
        print("[ImageBatchStream] Processing ", f)
        img = ImageBatchStream.read_image_chw(f)
        imgs.append(img)
        #import pdb; pdb.set_trace()
      for i in range(len(imgs)):
        self.calibration_data[i] = imgs[i]
      self.batch += 1
      return np.ascontiguousarray(self.calibration_data, dtype=np.float32)
    else:
      return np.array([])
    
def create_calibration_dataset(dataset_loc, num_samples):
  # Create list of calibration images 
  # This sample code picks n images at random for each object
  sampled_images = []
  for scene in os.listdir(dataset_loc):
    #00001,00002
    images = glob.glob(os.path.join(dataset_loc,scene)+"/rgb/*.png")
    #print(os.path.join(dataset_loc,scene))
    shuffle(images)
    sampled_images.extend(images[:num_samples])

  return sampled_images

if __name__ == "__main__":

  print('Calibration image size:', args.img_size)
  print('Number of samples that will be selected for every object:', args.num_samples)
  print('Number of samples that will be processed in every iteration (batch size):', args.batch_size)

  builder = trt.Builder(TRT_LOGGER)
  network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
  parser = trt.OnnxParser(network, TRT_LOGGER)

  # parse ONNX
  with open(args.onnx, 'rb') as model:
    print('Beginning ONNX file parsing')
    parser.parse(model.read())
  print('Completed parsing of ONNX file')

  calibration_files = create_calibration_dataset(args.calib_dataset_loc, args.num_samples)
  print('Num of calibration files:', len(calibration_files))
  config = builder.create_builder_config()

  # Process batch_size images at a time for calibration
  # This batch size can be different from MaxBatchSize (1 in this example)
  batchstream = ImageBatchStream(args.batch_size, calibration_files) # was 64
  Int8_calibrator = PythonEntropyCalibrator(batchstream,args.saveCache)
  
  config.set_flag(trt.BuilderFlag.INT8)
  config.int8_calibrator = Int8_calibrator

  serialized_engine = builder.build_serialized_network(network, config)
  with open("sample.engine", "wb") as f:
     f.write(serialized_engine)

  images_not_used_for_calib = [f for f in glob.glob(args.calib_dataset_loc+"/*.png") if f not in calibration_files]
  print(images_not_used_for_calib)