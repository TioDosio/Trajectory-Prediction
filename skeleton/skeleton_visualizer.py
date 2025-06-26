#!/usr/bin/env python
import rospy
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from human_awareness_msgs.msg import PersonsList


def publish_body_parts(persons_msg):
    marker_array = MarkerArray()
    marker_id = 0

    for person_idx, person in enumerate(persons_msg.persons):
        for bp in person.body_parts:
            marker = Marker()
            marker.header = persons_msg.header
            marker.ns = "body_parts"
            marker.id = marker_id
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            
            # Convert pixel coordinates to reasonable world coordinates
            # Scale down pixel values to meters (adjust scale factor as needed)
            marker.pose.position.x = bp.x / 1000.0  # Convert pixels to meters
            marker.pose.position.y = bp.y / 1000.0  # Convert pixels to meters
            marker.pose.position.z = 0.1  # Slightly above ground
            
            # Set proper orientation (required for valid quaternion)
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0
            
            marker.scale.x = 0.05
            marker.scale.y = 0.05
            marker.scale.z = 0.05
            marker.color.a = 1.0
            
            # Clamp color values to valid range [0.0, 1.0]
            r_val = 1.0 - bp.confidence
            g_val = bp.confidence
            
            marker.color.r = max(0.0, min(1.0, r_val))
            marker.color.g = max(0.0, min(1.0, g_val))
            marker.color.b = 0.0
            
            marker.lifetime = rospy.Duration(0.5)  # Markers disappear if not updated

            marker_array.markers.append(marker)
            marker_id += 1

    marker_pub.publish(marker_array)

if __name__ == "__main__":
    rospy.init_node("body_parts_visualizer")
    marker_pub = rospy.Publisher("/skeleton_markers", MarkerArray, queue_size=1)

    def persons_callback(msg):
        publish_body_parts(msg)

    rospy.Subscriber("/image_detections", PersonsList, persons_callback)
    rospy.spin()
