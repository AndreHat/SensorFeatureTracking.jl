# First compile all functions in RT_tracking_functions_list.jl
#


## Section 1: import packages

using Images, ImageView, ImageDraw, ImageFeatures, Gtk.ShortNames, VideoIO, ImageFiltering
using SensorFeatureTracking
using ProfileView

#######
## Section 2: Setup structs and use Harris corner detection to create a list of features

const TrackingType_Forward = "Forward"
const TrackingType_Pyramid = "Pyramid"
const TrackingType_Inversed = "Inversed"
const TrackingType_InversedPyramid = "InversedPyramid"

const windowSize_25 = 25            # this is half the size of the window to look for a matching feature. Actual window size is windowSize * 2 + 1
const windowSize_20 = 20            # windowSize = windowSize_20 will result in a 41x41 window
const windowSize_15 = 15
const windowSize_10 = 10
windowSize = 20

frame_counter_from = 50            # in an image sequence you can select a range of images from frame_counter_from + 1 to frame_counter_to
frame_counter_to = 98
nfeatures=20

frame_counter = 4
window_counter = 0
update_reference_frame_count = 4

number_frames = frame_counter_to - frame_counter_from +1



#------------------------------------------------------------------------------------------------------

img = load(joinpath(dirname(@__FILE__),"../Data/testSequence/image_$(frame_counter_from ).jpg"))
img_end = load(joinpath(dirname(@__FILE__),"../Data/testSequence/image_$(frame_counter_to).jpg"))

#------------------------------------------------------------------------------------------------------




I = Gray.(img);
orgI_setup = deepcopy(img)
orgI_setup = Gray.(orgI_setup)
orgI_setup = convert(Array{Float64}, orgI_setup)
orgI_setup
img_end
#harris corner detection=============

# corners = getApproxBestHarrisInWindow(orgI_setup, nfeatures=nfeatures)
orgI_corners = orgI_setup[windowSize:length(orgI_setup[:,1])-windowSize , windowSize:length(orgI_setup[1,:])-windowSize]
corners = getApproxBestShiTomasi(orgI_corners, nfeatures=nfeatures)   # function in RT_tracking_functions_list.jl
number_features_setup = length(corners[:,1])
# for count = 1:number_features_setup
#     corners[1].keypoint = corners[1].keypoint.+windowSize_25
# end
# corners[:].Keypoints
#harris corner detection=============

# images used to draw and display the tracked feature's path
feature_path_img_start = deepcopy(img)
feature_path_img_start = Gray.(feature_path_img_start);

feature_path_img_end = deepcopy(img_end)
feature_path_img_end = Gray.(feature_path_img_end);

#constructor function for structs ITVar and ITConst
ITVar, ITConst = ImageTrackerSetup(orgI_setup, corners, windowSize = windowSize, TrackingType_setup = TrackingType_Inversed)
# fillNewImageTemplates!(ITVar, ITConst)
#########
# Section 3: Run KLT tracker on the selected image sequence
Profile.clear()
frame_counter = 2
window_counter = 0
# p_reference_backup::Array{CartesianIndex{2}}
p_reference_backup = ITVar.p_reference
NextFrame = zeros(length(ITVar.orgI[:,1]),length(ITVar.orgI[1,:]))
I_nextFrame = zeros(length(ITVar.orgI[:,1]),length(ITVar.orgI[1,:]))
tic()
while frame_counter <= number_frames
    @show frame_counter
    window_counter += 1
    # Load next image
    #------------------------------------------------------------------------------------------------------
    NextFrame[:,:] = load(joinpath(dirname(@__FILE__),"../Data/testSequence/image_$(frame_counter + frame_counter_from ).jpg"))
    #------------------------------------------------------------------------------------------------------

    NextFrame[:,:] = Gray.(NextFrame)

    NextFrame[:,:] = convert(Array{Float64}, NextFrame)
    I_nextFrame[:,:] = convert(Array{Float64}, NextFrame)

    ITVar.I_nextFrame[:,:] = I_nextFrame

    if (ITConst.TrackingType == "Pyramid" || ITConst.TrackingType == "InversedPyramid")
        ITVar.I_nextFrame_downsample[:,:] = imresize(I_nextFrame, (Int(length(I_nextFrame[:,1])/ITConst.downsampleFactor),Int(length(I_nextFrame[1,:])/ITConst.downsampleFactor)));
    end
    # ImageView.imshow(ITVar.I_nextFrame[:,:])
    # ImageView.imshow(NextFrame)
        # Main KLT tracker function ==============
        # @time KTL_Tracker!(ITVar, ITConst)
        KTL_Tracker!(ITVar, ITConst)
        # @profile KTL_Tracker!(ITVar, ITConst)
        # @code_warntype KTL_Tracker!(ITVar, ITConst)
        # Main KLT tracker function ==============

    # Draw the path of the feature on both the first and last image in the sequence
    # This code draws lines between the reference frame and the tracked feature and not sequential frames.
    # If you need a frame to frame feature path set update_reference_frame_count = 1, however this will decrease the accuracy of the tracker
        for columnCount = 1:length(ITVar.p_reference[1,:])
            if (ITVar.p_reference[2, columnCount][1] <= 0   ||   ITVar.p_reference[2, columnCount][2] <= 0)
                # @show fail=1
            else
                # draw!(feature_path_img_start, LineSegment(ITVar.p_reference[1, columnCount],ITVar.p_reference[2, columnCount]))
                draw!(feature_path_img_start, LineSegment(p_reference_backup[2, columnCount],ITVar.p_reference[2, columnCount]))
                p_reference_backup .= ITVar.p_reference
            end
        end
        for columnCount = 1:length(ITVar.p_reference[1,:])
            if (ITVar.p_reference[2, columnCount][1] <= 0   ||   ITVar.p_reference[2, columnCount][2] <= 0)
                # @show fail=1
            else
                # draw!(feature_path_img_end, LineSegment(ITVar.p_reference[1, columnCount],ITVar.p_reference[2, columnCount]))
                draw!(feature_path_img_end, LineSegment(p_reference_backup[2, columnCount],ITVar.p_reference[2, columnCount]))
                p_reference_backup .= ITVar.p_reference
            end
        end

    # Update the frame used as reference for tracking
    # Making update_reference_frame_count larger reduces feature drift but can make the tracker lose the feature if the feature change to much
    if (window_counter == update_reference_frame_count)
        # if frame_counter == 96
        #     @show ITVar
        # end
        # updateFeatures!(ITVar, ITConst)
        # fillNewImageTemplates!(ITVar, ITConst)
        if (addBestShiTomasi!(ITVar, ITConst, halfWindowSize = 9))
          # fillNewImageTemplates!(ITVar, ITConst)
          @show updated = "updated+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
        end

        window_counter = 0
    end

    frame_counter += 1
end
toc()
ImageView.imshow(feature_path_img_start)
ImageView.imshow(feature_path_img_end)
Atom.Profiler.profiler()
####
