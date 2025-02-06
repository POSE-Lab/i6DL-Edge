import glob 
from absl import logging,flags,app
import os
import shutil
#from natsort import natsorted 

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'srv_res_path',"",
    'Service result path, \
    usually found in the folder /root/sanbox/results/XXXXX/image_raw'
)

flags.DEFINE_string(
    'output_folder','',
    'Path the visualizations will be kept'

)

def main(arg):
    logging.info(FLAGS.output_folder)
    if not os.path.exists(FLAGS.output_folder):
        os.makedirs(FLAGS.output_folder)

    folders = glob.glob(FLAGS.srv_res_path+"/*")
    logging.info(len(folders))
    for f in folders:
        objid = glob.glob(f+"/*")[0]
        img_path = os.path.join(f, objid, "result.png")
        logging.info(img_path)
        new_img_name = f.split("/")[-1]+"_result.png"
        new_img_path = os.path.join(FLAGS.output_folder, new_img_name)
        logging.info(new_img_path)
        #if not os.path.exists(new_img_path):
        logging.info('Copying ' +str(img_path)+' to '+str(new_img_path))
        shutil.copy2(img_path, new_img_path)


if __name__=="__main__":
    app.run(main)
