import numpy as np
import cv2

import os

# I/O directories
input_dir = "input"
output_dir = "output"



class ParticleFilter(object):
    """A particle filter tracker, encapsulating state, initialization and update methods."""

    currentParticleSet = np.empty((0,3))
    patchTemplate = None
    def __init__(self, frame, template, **kwargs):

        self.tracker = cv2.imread('output/tracker.png')
        self.patchTemplate = np.array(template,dtype=np.int16)
        self.num_particles = kwargs.get('num_particles', 400)  # extract num_particles (default: 100)
        self.ppa= self.num_particles**.5 # particles per axis
        self.model_dynamics_sd = kwargs.get('model_dynamics_gaussian',10)
        self.MSE_sigma = kwargs.get('MSE_sigma',10)
        # TODO: Your code here - extract any additional keyword arguments you need and initialize state

    def calculateMSE(self,frame):
        for indx,center in enumerate(self.currentParticleSet):
            center_row = center[0]
            center_column = center[1]
            h,w = self.patchTemplate.shape
            imgPatch = frame[center_row-h/2:center_row+h/2,center_column-w/2:center_column+w/2]
            img_Patch_temp = np.zeros(self.patchTemplate.shape)
            img_Patch_temp[0:imgPatch.shape[0],0:imgPatch.shape[1]] = imgPatch
            MSE = np.sum((self.patchTemplate - img_Patch_temp)**2)/(h*w)
            MSE_measurable = np.e**(-MSE/(2*(self.MSE_sigma**2)))
            self.currentParticleSet[indx,2] = MSE_measurable
        MSE_Sum = self.currentParticleSet[:,2].sum()
        self.currentParticleSet[:,2] = self.currentParticleSet[:,2]/MSE_Sum

    def resampleParticles(self):
        weights = self.currentParticleSet[:,2]
        n = len(weights)
        indices = []
        C = [0.] + [sum(weights[:i+1]) for i in range(n)]
        u0, j =  np.random.random(), 0
        for u in [(u0+i)/n for i in range(n)]:
          while u > C[j]:
            j+=1
          indices.append(j-1)
        self.currentParticleSet = self.currentParticleSet[indices]


    def process(self, frame):
        """Process a frame (image) of video and update filter state.
        """
        # TODO: Your code here - use the frame as a new observation (measurement) and update model

        if(len(self.currentParticleSet) == 0):
            xset = np.linspace(52,1220,self.ppa).astype("int")
            yset = np.linspace(65,650,self.ppa).astype("int")
            for i in range(int(self.ppa)):
                for j in range(int(self.ppa)):
                    self.currentParticleSet = np.append(self.currentParticleSet,[[yset[i],xset[j],1.0/self.num_particles]],axis=0)

        self.calculateMSE(frame)




    def render(self, frame_out):
        """Visualize current particle filter state.
        """
        h= self.patchTemplate.shape[0]
        w= self.patchTemplate.shape[1]
        center_x = int(np.mean(self.currentParticleSet[:,1]))
        center_y = int(np.mean(self.currentParticleSet[:,0]))
        cv2.rectangle(frame_out,(int(center_x-w/2),int(center_y-h/2)),(int(center_x+w/2),int(center_y+h/2)),(0,255,0),1)


        # cv2.rectangle(self.tracker, (int(center_x - w / 2), int(center_y - h / 2)),
        #               (int(center_x + w / 2), int(center_y + h / 2)), 127, 1)

        cv2.circle(self.tracker, (int(center_x), int(center_y)), 5, (255,0,0),-1)

        # Drawing the particles
        for indx,particle in enumerate(self.currentParticleSet):
            cv2.circle(frame_out,(int(particle[1]),int(particle[0])),1,(0,255,0),1)

        # Computing the weighted centerx and centery
        center_weighted_x = np.sum(self.currentParticleSet[:,1]*self.currentParticleSet[:,2])
        center_weighted_y = np.sum(self.currentParticleSet[:,0]*self.currentParticleSet[:,2])

        # Computing the weighted radius
        weighted_radius = (((((self.currentParticleSet[:,0:2]-np.array([[center_weighted_y,center_weighted_x]]))**2).sum(axis=1))**.5)*self.currentParticleSet[:,2]).sum()

        #Drawing the circle with the weighted radius
        cv2.circle(frame_out,(int(center_weighted_x),int(center_weighted_y)),int(weighted_radius),(0,0,255),2)
        return frame_out, self.tracker


class AppearanceModelPF(ParticleFilter):
    """A variation of particle filter tracker that updates its appearance model over time."""

    def __init__(self, frame, template, **kwargs):
        """Initialize appearance model particle filter object (parameters same as ParticleFilter)."""
        super(AppearanceModelPF, self).__init__(frame, template, **kwargs)  # call base class constructor
        # TODO: Your code here - additional initialization steps, keyword arguments
        self.alpha = kwargs.get('alpha',.5)
        x,y,w,h = 529,375,98,132
        self.templateCoords = kwargs.get('templateCoords',dict({'x':x,'y':y,'w':w,'h':h}))
    # TODO: Override process() to implement appearance model update
    def process(self, frame):
        """Process a frame (image) of video and update filter state.

        Parameters
        ----------
            frame: color BGR uint8 image of current video frame, values in [0, 255]
        """
        # TODO: Your code here - use the frame as a new observation (measurement) and update model

        if(len(self.currentParticleSet) == 0):
            xset = np.linspace(52,1220,self.ppa).astype("int")
            yset = np.linspace(65,650,self.ppa).astype("int")
            for i in range(int(self.ppa)):
                for j in range(int(self.ppa)):
                    self.currentParticleSet = np.append(self.currentParticleSet,[[yset[i],xset[j],1.0/self.num_particles]],axis=0)

        self.calculateMSE(frame)
        self.update_template_iir(frame)

    def update_template_iir(self,frame):
        center_weighted_x = np.sum(self.currentParticleSet[:,1]*self.currentParticleSet[:,2])
        center_weighted_y = np.sum(self.currentParticleSet[:,0]*self.currentParticleSet[:,2])

        h,w = self.patchTemplate.shape
        frame_y_range = -int(h/2)

        new_window = frame[int(center_weighted_y-int(h/2)):int(center_weighted_y+int(h/2)),int(center_weighted_x-int(w/2)):int(center_weighted_x+int(w/2))]

        # self.templateCoords['x'] = int(self.alpha*(center_weighted_x-int(w/2)) + (1-self.alpha) * self.templateCoords['x'])
        # self.templateCoords['y'] = int(self.alpha*(center_weighted_y-int(h/2)) + (1-self.alpha) * self.templateCoords['y'])
        # self.patchTemplate = frame[self.templateCoords['y']:self.templateCoords['y']+self.templateCoords['h'],self.templateCoords['x']:self.templateCoords['x']+self.templateCoords['w']]

        window_patch_size = np.zeros((self.templateCoords['h'],self.templateCoords['w']))
        hx = 0
        hy = 0
        if(new_window.shape[0] > self.templateCoords['h']):
            hy=self.templateCoords['h']
        else:
            hy = new_window.shape[0]

        if(new_window.shape[1] > self.templateCoords['w']):
            hx=self.templateCoords['w']
        else:
            hx = new_window.shape[1]


        window_patch_size[0:hy,0:hx] = new_window[0:hy,0:hx]
        self.patchTemplate = self.alpha * window_patch_size + (1 - self.alpha) * self.patchTemplate;




    # TODO: Override render() if desired (shouldn't have to, ideally)





# Driver/helper code
def get_template_rect(rect_filename):
    """Read rectangular template bounds from given file.

    The file must define 4 numbers (floating-point or integer), separated by whitespace:
    <x> <y>
    <w> <h>

    Parameters
    ----------
        rect_filename: path to file defining template rectangle

    Returns
    -------
        template_rect: dictionary specifying template bounds (x, y, w, h), as float or int

    """
    with open(rect_filename, 'r') as f:
        values = [float(v) for v in f.read().split()]
        return dict(zip(['x', 'y', 'w', 'h'], values[0:4]))


def run_particle_filter(pf_class, video_filename, template_rect, save_frames={}, **kwargs):


    # Open video file
    video = cv2.VideoCapture(video_filename)

    w, h = video.get(3), video.get(4)
    tracker_array = np.zeros((h, w))
    cv2.imwrite('output/tracker.png', tracker_array )
    # Initialize objects
    template = None
    pf = None
    frame_num = 0
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    videoout = cv2.VideoWriter('video.avi',fourcc,fps=20,frameSize=(1280,720))
    videoout_tracker = cv2.VideoWriter('video_tracker.avi', fourcc, fps=20, frameSize=(1280, 720))
    framecount =0
    # Loop over video (till last frame or Ctrl+C is presssed)
    while True:
        try:
            framecount += 1
            # Try to read a frame
            okay, frame = video.read()
            if not okay:
                break  # no more frames, or can't read video
            frame_GRAY = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            # Extract template and initialize (one-time only)
            if template is None:
                template = frame_GRAY[int(template_rect['y']):int(template_rect['y'] + template_rect['h']),
                                 int(template_rect['x']):int(template_rect['x'] + template_rect['w'])]
                if 'template' in save_frames:
                    cv2.imwrite(save_frames['template'], template)
                pf = pf_class(frame_GRAY, template, **kwargs)

            # Process frame
            pf.process(frame_GRAY)  # TODO: implement this!

            frame_out = frame.copy()
            frame_out, frame_tracker = pf.render(frame_out)
            videoout.write(frame_out)
            videoout_tracker.write(frame_tracker)

            # Render and save output, if indicated
            if frame_num in save_frames:
                cv2.imwrite(save_frames[frame_num], frame_out)

            # Update frame number
            frame_num += 1

            # Resample particles & add model dynamics
            pf.resampleParticles()


            # adding model dynamics
            X = np.random.normal(0,pf.model_dynamics_sd,pf.ppa**2).astype(np.int)
            Y = np.random.normal(0,pf.model_dynamics_sd,pf.ppa**2).astype(np.int)
            pf.currentParticleSet[:,0]+=Y
            pf.currentParticleSet[:,1]+=X
            pf.currentParticleSet[:,2] = 1/pf.num_particles
            if(framecount == 141):
                break


        except KeyboardInterrupt:  # press ^C to quit
            break
    videoout.release()
    videoout_tracker.release()


def main():
    templatedict = get_template_rect(os.path.join(input_dir, "DRS.txt"))
    run_particle_filter(AppearanceModelPF,
        os.path.join(input_dir, "DRS.mp4"),
        templatedict,
        {
            'template': os.path.join(output_dir, 'ps7-2-a-1.png'),
            15: os.path.join(output_dir, 'ps7-2-a-2.png'),
            50: os.path.join(output_dir, 'ps7-2-a-3.png'),
            140: os.path.join(output_dir, 'ps7-2-a-4.png')
        },
        num_particles=900,model_dynamics_gaussian=20,MSE_sigma=1,alpha=.3,templateCoords = templatedict)  # TODO: Tune parameters so that model can continuing tracking through noise



if __name__ == "__main__":
    main()
