import rospy 
from odl.msg import ObjectPose

def handle_callback(data):
    rospy.loginfo(rospy.get_caller_id() + "Data recieved at " + str(rospy.get_time()))

def listener():

    rospy.init_node('listener',anonymous=True)

    rospy.Subscriber("/ICCS/ObjectDetectionAndLocalization/ObjectPose",ObjectPose,handle_callback)

    rospy.spin()

if __name__ == "__main__":
    listener()