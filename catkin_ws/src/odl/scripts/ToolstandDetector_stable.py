import os
import yaml
import cv2 as cv
import numpy as np
from absl import flags,app,logging
from numpy.typing import ArrayLike,NDArray
from utilsHF import rotate3D_X,rotate3D_Y,match_inter_board_points
import typing as T


class ToolstandDetector():
    """
    Class incorporating the functionality of the toolstand detection.

    Given a predifiend aruco dictionary the class creates the correspoding ArUco board.
    Then when the detectArucoMarkers() method is called 
    """
    def __init__(self,config_file:T.Union[os.PathLike,bytes,str]) -> None:
        
        # load configurtation file into class attributes
        with open(config_file,'r') as f:
            attrs = yaml.load(f,yaml.FullLoader)
            for key in attrs:
                print(key,attrs[key])
                setattr(self, key, None if attrs[key] == 'None' else attrs[key])
        #self.model = load3DModels(self.toolstand_model_path)[0]

        self.aruco_dict= cv.aruco.getPredefinedDictionary(eval("cv.aruco." + str(self.ArUco_dict)))
        self.board = cv.aruco.GridBoard((self.grid_board['rows'],
                                   self.grid_board['columns']),
                                   self.grid_board['mrk_size'],
                                   self.grid_board['mrk_space'],
                                   self.aruco_dict)
    @staticmethod
    def detectArucoMarkers(arucodict: cv.aruco.Dictionary,image: NDArray) -> list:
        """Given an predefined ArUco dictionary and an RGB image performs 
        marker detection. 

        Args:
            arucodict (cv.aruco.Dictionary): _description_
            image (NDArray): An rgb image in the form of a numpy array

        Returns:
            tuple[ArrayLike,ArrayLike,ArrayLike]: Returns the detected 
            2D corner coordinates, the detected markers ids and the rejected markers id's that
            have been detected as markers but have been rejected during the verification
            proccess.
        """
        arucoDict = cv.aruco.getPredefinedDictionary(arucodict)
        arucoParams = cv.aruco.DetectorParameters()
        detector = cv.aruco.ArucoDetector(arucoDict,arucoParams)

        return detector.detectMarkers(image)

    def calc_confidence(self,original_image_points,object_points,rvec,tvec,K,WS_id,num_detected,distCoeffs=None,repr_threshold= 2): # num_detected new

        assert original_image_points.shape[0] == object_points.shape[0], \
        f"Shapes {original_image_points.shape} and {object_points} do not match"

        reprojected,_ = cv.projectPoints(object_points,rvec,tvec,cameraMatrix=K,distCoeffs=distCoeffs)

        rerror = np.linalg.norm((original_image_points - reprojected).reshape(-1,2),axis=1)

        accepted = np.count_nonzero(rerror[rerror < repr_threshold])

        print(f"Number of detected markers is : {num_detected}")
        print(f"Accepted: {accepted}")
        num_markers_total = 50 if WS_id=="10" else num_detected # it was 18 , num_detected new
        self.score = accepted/(num_markers_total * 4)
        
        print(accepted,(len(original_image_points) * 4))
        print(f"Total score: {self.score}")
        return self.score
    
    def detect(self,rgb,WS_id):
        
        num_detected = []
        WS_id = self.toolstand_map[WS_id]
        print("WORKSTATION ID: ",WS_id)
        self.rgb = rgb # might need cv.cvtColor(rgb,cv.COLOR_BGR2RGB)
        print(self.K)
        self.corners,self.ids,_ = self.detectArucoMarkers(eval("cv.aruco." + str(self.ArUco_dict)),
                                                rgb)
        if self.corners[0].size >= self.detected_marker_threshold:
            
            logging.debug(f"Number of corners detected: {self.corners[0].size}")
            if "2" in WS_id:
                BOARD_1 = cv.aruco.GridBoard((3,3),48,10,self.aruco_dict,ids=np.arange(0,9))
                BOARD_2 = cv.aruco.GridBoard((3,3),48,10,self.aruco_dict,ids=np.arange(9,18))

                if WS_id == "21":
                    num_detected = [id for id in self.ids if id in np.arange(0,9)] # new
                elif WS_id == "22":
                    num_detected = [id for id in self.ids if id in np.arange(9,18)] # new
                
                valid_corners = np.array([self.corners[i] for i, id in enumerate(self.ids) if id in num_detected])
                valid_ids = np.array([id for id in self.ids if id in num_detected])

                if WS_id == "21":
                    object_points,image_points = BOARD_1.matchImagePoints(valid_corners,valid_ids)
                    object_points += self.UL_CORNER_TO_LEFT_TOOL_HOLDER
                else:
                    object_points,image_points = BOARD_2.matchImagePoints(valid_corners,valid_ids)
                    object_points += self.RIGHT_TOOL_HOLDER_TO_LEFT_TOOL_HOLDER #TODO
            else:
                object_points,image_points = self.board.matchImagePoints(self.corners,self.ids)
                # convert the object point coordinates to desired coordinate frame (e.g. with respect to some new origin)
                object_points += np.array(self.UL_MARKER_TO_TOOLSTAND_T)[None,:]
            


            _,self.rvec,self.tvec = cv.solvePnP(object_points, 
                                      image_points, 
                                      cameraMatrix=np.array(self.K).reshape(3,3), 
                                      distCoeffs=None)

            # get the correspoding 3x3 rotation matrix
            rot_mat,_ = cv.Rodrigues(self.rvec)
            
            # obtain the final pose
            rt_final = np.eye(4)
            rt_final[:3,:3] = rot_mat.copy() @ rotate3D_X(-90) @ rotate3D_Y(-90)
            rt_final[:-1,-1] = self.tvec.reshape(1,3)
            
            self.predicted_pose = rt_final
            logging.debug(f"Final pose: {rt_final}")
        else:
            logging.fatal("[ToolstandDetector::detect] Too few markers detected, try altering the camera viewpoint")
        
        print(f"Len image points: {image_points.shape} , Len object_points: {object_points.shape}")
        return rt_final, self.calc_confidence(image_points,object_points,self.rvec,self.tvec,np.array(self.K).reshape(3,3),WS_id,len(num_detected),repr_threshold = self.reprojection_error_threshold) # num_detected new
    
    
    def visualize(self,savePath, imgname):

        K = np.array(self.K).reshape(3,3)
        cv.drawFrameAxes(self.rgb,K,None,self.predicted_pose[:3,:3],self.predicted_pose[:-1,-1],length=80)
        cv.aruco.drawDetectedMarkers(self.rgb, self.corners,self.ids)
        cv.imwrite(os.path.join(savePath,imgname +".png"),self.rgb)


