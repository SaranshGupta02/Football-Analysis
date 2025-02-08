from ultralytics import YOLO
import supervision as sv
import pickle
import os
import numpy as np
import pandas as pd
import cv2
import sys 
sys.path.append('../')
from utils.bbox_utils import get_center_of_bbox, get_bbox_width, get_foot_position

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path) 
        self.tracker = sv.ByteTrack()
   
    def add_position_to_tracks(sekf,tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    if object == 'ball':
                        position= get_center_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                    tracks[object][frame_num][track_id]['position'] = position
    

    def interpolate_ball_positions(self,ball_positions):
        
        ball_positions = [x.get(1,{}).get('bbox',[]) for x in ball_positions] # track id 1 mtlb ball agar ball ni hui toh empty dict and ball ki bbox
        df_ball_positions = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])

        #jaise hm video dekh rhe hh abhi toh kuch frames m ball ni aahrhi toh isme vo null honge toh unme predict krke bbox bnane h 
        # Interpolate missing values
        
        df_ball_positions = df_ball_positions.interpolate(method='linear')# as ball approx linear path followkregi
        df_ball_positions = df_ball_positions.bfill()# agr ab bhi null bachi toh fill krdo

        ball_positions = [{1: {"bbox":x}} for x in df_ball_positions.to_numpy().tolist()] #purane format me krdi

        return ball_positions

    def detect_frames(self, frames):
        batch_size=20 
        detections = [] 
        for i in range(0,len(frames),batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size],conf=0.1)
            detections += detections_batch
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path,'rb') as f:
                tracks = pickle.load(f)
            return tracks

        detections = self.detect_frames(frames)
    #yeh aisa hoga tracks["players"][i] will be {playerid:{bbox:{}} mtlb dekh hr frame ke liye hr player(ki track id aur uska bbox h )
        tracks={
            "players":[],
            "referees":[],
            "ball":[]
        }

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v:k for k,v in cls_names.items()}

            # Covert to supervision Detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            #dataset km tha toh goalkeeper aur player m confuse ho rha h hence yeh kre
            for object_ind , class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_ind] = cls_names_inv["player"]
                    #detection_supervision m req format m convert kra h hmne

            # Track Objects
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)
            #print(detection_with_tracks) dekh isme  track id bhi h mt ab detection_with_trac[frame] m hr identified object ka bbox class id track id h 
            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv['player']:
                    tracks["players"][frame_num][track_id] = {"bbox":bbox}
                
                if cls_id == cls_names_inv['referee']:
                    tracks["referees"][frame_num][track_id] = {"bbox":bbox}
            
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv['ball']:
                    tracks["ball"][frame_num][1] = {"bbox":bbox}

        if stub_path is not None:
            with open(stub_path,'wb') as f:
                pickle.dump(tracks,f)

        return tracks
    
    def draw_ellipse(self,frame,bbox,color,track_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center,y2),
            axes=(int(width), int(0.35*width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color = color,
            thickness=2,
            lineType=cv2.LINE_4
        )
        #rectangle jo draw ho rh ah jisme num likh ke aarha
        rectangle_width = 40
        rectangle_height=20
        x1_rect = x_center - rectangle_width//2
        x2_rect = x_center + rectangle_width//2
        y1_rect = (y2- rectangle_height//2) +15
        y2_rect = (y2+ rectangle_height//2) +15

        if track_id is not None:
            cv2.rectangle(frame,
                          (int(x1_rect),int(y1_rect) ),
                          (int(x2_rect),int(y2_rect)),
                          color,
                          cv2.FILLED)
            
            x1_text = x1_rect+12
            if track_id > 99:
                x1_text -=10 # wrna text bhr nikaljaega
            
            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text),int(y1_rect+15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,0,0),
                2
            )

        return frame

    def draw_traingle(self,frame,bbox,color):
        y= int(bbox[1])
        x,_ = get_center_of_bbox(bbox)

        triangle_points = np.array([
            [x,y],
            [x-10,y-20],
            [x+10,y-20],
        ])
        cv2.drawContours(frame, [triangle_points],0,color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points],0,(0,0,0), 2)

        return frame

    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
    # Draw a semi-transparent rectangle at the bottom for better visibility
        overlay = frame.copy()
        cv2.rectangle(overlay, (800, 610), (1225, 700), (255, 255, 255), -1)  # White background
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Calculate ball control percentages
        team_ball_control_till_frame = team_ball_control[:frame_num + 1]
        team_1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame == 1].shape[0]
        team_2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame == 2].shape[0]

        if team_1_num_frames + team_2_num_frames > 0:
            team_1 = team_1_num_frames / (team_1_num_frames + team_2_num_frames)
            team_2 = team_2_num_frames / (team_1_num_frames + team_2_num_frames)
        else:
            team_1, team_2 = 0, 0  # Avoid division by zero

        # Define text and positions
        text1 = f"Team 1 Ball Control: {team_1 * 100:.2f}%"
        text2 = f"Team 2 Ball Control: {team_2 * 100:.2f}%"
        pos1 = (810, 650)
        pos2 = (810, 680)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2

        cv2.putText(frame, text1, pos1, font, font_scale, (0, 0, 0), thickness)
        cv2.putText(frame, text2, pos2, font, font_scale, (0, 0, 0), thickness)

        return frame

    def draw_annotations(self,video_frames, tracks,team_ball_control):
        output_video_frames= []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]

            # Draw Players
            for track_id, player in player_dict.items():
                color = player.get("team_color",(0,0,255))
                frame = self.draw_ellipse(frame, player["bbox"],color, track_id)

                if player.get('has_ball',False):
                    frame = self.draw_traingle(frame, player["bbox"],(0,0,255))

            #  Draw Referee
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"],(0,255,255))
            
            # # Draw ball 
            for track_id, ball in ball_dict.items():
                 frame = self.draw_traingle(frame, ball["bbox"],(0,255,0))


            frame=self.draw_team_ball_control(frame,frame_num,team_ball_control)
            output_video_frames.append(frame)

        return output_video_frames