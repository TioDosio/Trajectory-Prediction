#!/usr/bin/env python
import rospy
from visualization_msgs.msg import Marker, MarkerArray
from human_awareness_msgs.msg import PersonsList, BodyPart
from geometry_msgs.msg import Point, Quaternion

class SkeletonVisualizer:
    def __init__(self):
        rospy.init_node('skeleton_visualizer')
        self.marker_pub = rospy.Publisher('/skeleton_markers', MarkerArray, queue_size=10)
        self.sub = rospy.Subscriber('/people', PersonsList, self.callback)
        self.show_labels = rospy.get_param('~show_labels', True)
        self.show_confidence = rospy.get_param('~show_confidence', False)
        
        # Define skeleton connections (modify based on your part_id values)
        self.connections = [
            ('Nose', 'Neck'), ('Neck', 'RShoulder'), ('RShoulder', 'RElbow'),
            ('RElbow', 'RWrist'), ('Neck', 'LShoulder'), ('LShoulder', 'LElbow'),
            ('LElbow', 'LWrist'), ('Neck', 'MidHip'), ('MidHip', 'RHip'),
            ('RHip', 'RKnee'), ('RKnee', 'RAnkle'), ('MidHip', 'LHip'),
            ('LHip', 'LKnee'), ('LKnee', 'LAnkle')
        ]

    def callback(self, msg):
        marker_array = MarkerArray()
        
        for i, person in enumerate(msg.persons):
            # Process body parts
            parts = {part.part_id: part for part in person.body_parts}
            
            # Create markers for each body part
            for part in person.body_parts:
                marker = Marker()
                marker.header = msg.header
                marker.ns = "person_{}".format(i)
                # Use hash and abs to ensure positive int id
                marker.id = abs(hash(part.part_id)) % 100000
                marker.type = Marker.SPHERE
                marker.action = Marker.ADD
                marker.pose.position.x = part.x
                marker.pose.position.y = part.y
                marker.pose.position.z = 0  # Adjust if you have Z coordinates
                marker.scale.x = 0.1
                marker.scale.y = 0.1
                marker.scale.z = 0.1
                marker.color.a = 1.0
                marker.color.r = 1.0 if 'R' in part.part_id else 0.0
                marker.color.g = 1.0 if 'L' in part.part_id else 0.0
                
                if self.show_confidence:
                    marker.color.b = part.confidence
                else:
                    marker.color.b = 0.0
                
                marker_array.markers.append(marker)

            # Create skeleton connections
            for (start, end) in self.connections:
                if start in parts and end in parts:
                    line = Marker()
                    line.header = msg.header
                    line.ns = "person_{}_lines".format(i)
                    line.id = abs(hash(start + end)) % 100000
                    line.type = Marker.LINE_STRIP
                    line.action = Marker.ADD
                    line.scale.x = 0.05
                    line.color.a = 0.8
                    line.color.r = 0.5
                    line.points.append(Point(parts[start].x, parts[start].y, 0))
                    line.points.append(Point(parts[end].x, parts[end].y, 0))
                    marker_array.markers.append(line)

        self.marker_pub.publish(marker_array)
        rospy.loginfo("Publishing {} markers".format(len(marker_array.markers)))

if __name__ == '__main__':
    SkeletonVisualizer()
    rospy.spin()
