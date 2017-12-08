using SensorFeatureTracking
using TransformUtils
using Base: Test
using PaddedViews
using Images
using Rotations
using CoordinateTransformations
using StaticArrays
using ImageDraw




function imrotate(I,angle,spec::String="central")

  tform = recenter(RotMatrix(angle/180*pi), center(I))
  Irot = parent(warp(I,tform))

  if (spec=="full")
    Irot
  elseif (spec=="central")
    szX,szY = size(I)
    Dx = Int64(round(szX/2))
    Dy = Int64(round(szY/2))
    x0 = Int64(round(center(Irot)[1]))
    y0 = Int64(round(center(Irot)[2]))
    Ifin = Irot[x0-Dx+1:x0-Dx+szX,y0-Dy+1:y0-Dy+szY]
  else
    I
  end

end

function padandcutoffsetImg(img, ro, co)
	blankImg = zeros(Gray{N0f8},ro,co)

	ro_start =  max(indices(img)[1].start, 1)
	ro_stop  = 	min(indices(img)[1].stop, ro)

	co_start =  max(indices(img)[2].start, 1)
	co_stop  = 	min(indices(img)[2].stop, co)

  blankImg[ro_start:ro_stop,co_start:co_stop] = img[ro_start:ro_stop,co_start:co_stop]

	return blankImg

end



function createImageSequence(imSequenceLength, noise)
  x= rand([0.0,1.0],(500,500))
  orgI = zeros(Float64, 500, 500)
  blank = zeros(Float64, 500, 500)
  warpedImg = ones(Float64, 500, 500)
  blankstack = Array{Float64,3}(500,500,imSequenceLength)

  orgI[193:210,193:210] = 0.2
  orgI[193:205,193:205] = 0.5
  orgI[193:201,193:201] = 0.7
  orgI[193:197,193:197] = 1.0
  orgI[195:210,195:210] = 0.0

  W_p = AffineMap(MMatrix{2,2}([1.0 0.0; -0.0 1.0]),MVector{2}([250,250]))
  if (noise==1)
    blank .=  orgI .- 0.1 .* x
  else
    blank .=  orgI
  end
  blankstack[:,:,1] .= blank

  for count = 2:imSequenceLength
    warped = warp(blank, W_p, 0.0)
    warpedImg = PaddedView(0.0, warped, (-249:250, -249:250))
    blank .= parent(warpedImg[:,:])
    blankstack[:,:,count] .= blank
  end
  return blankstack
end

function createImageSequence2(imSequenceLength, noise)
  x= rand([0.0,1.0],(500,500))
  orgI = zeros(Float64, 500, 500)
  blank = zeros(Float64, 500, 500)
  warpedImg = ones(Float64, 500, 500)
  blankstack = Array{Float64,3}(500,500,imSequenceLength)

  range = -25:25
  rangein = -20:20
  centres = [[125, 125], [325, 125], [225, 225], [325, 425], [125, 325], [125, 425]]
  intensities = [120, 180, 100, 200, 120, 150]
  testim = zeros(Float64,500,500)
  foreach((c,i) -> testim[c[1] + range, c[2] + range] = i/255, centres, intensities)
  foreach((c,i) -> testim[c[1] + rangein, c[2] + rangein] = (i/2)/255, centres, intensities)
  testim


  orgI .= testim

  # rtfm = recenter(RotMatrix(pi/150), SVector{2}(200, 200))
  # otfm = AffineMap(MMatrix{2,2}([1.004 0; -0.001 1.004]), MVector{2,Float32}(-1,0))
  rtfm = recenter(RotMatrix(pi/500), SVector{2}(200, 200))
  otfm = AffineMap(MMatrix{2,2}([1.005 0; -0.001 1.005]), MVector{2,Float32}(-1,0))


  W_p = AffineMap(MMatrix{2,2}(eye(2)), MVector{2,Float32}(0.,0))

  if (noise==1)
    orgI .=  orgI .- 0.1 .* x
  else
    orgI .=  orgI
  end

  for count = 1:imSequenceLength
    W_p = W_p ∘ rtfm ∘ otfm
    warped = warp(orgI, W_p, 0.0)
    warpedImg = PaddedView(0.0, warped, (1:500, 1:500))
    blank .= parent(warpedImg[:,:])
    blankstack[:,:,count] .= blank
  end
  return blankstack
end

##############


# ImageView.imshow(x)

@testset begin



nfeatures=25
imSequenceLength = 200
windowSize = 30
x= rand([0.0,1.0],(500,500))
orgI = zeros(Float64, 500, 500)
blankstack2 = Array{Float64,3}(500,500,imSequenceLength)

blankstack = createImageSequence2(imSequenceLength, 0)
# for i = 1:20
#   q = i*5
#   ImageView.imshow(blankstack[:,:,q])
# end

orgI .= blankstack[:,:,1]
drawimage = Gray.(deepcopy(orgI))

orgI_corners = orgI[windowSize:length(orgI[:,1])-windowSize , windowSize:length(orgI[1,:])-windowSize]

corners = getApproxBestShiTomasi(orgI_corners; nfeatures=nfeatures)


ITVar, ITConst = ImageTrackerSetup(orgI, corners, windowSize = windowSize, TrackingType_setup = "Inversed")


# blankstack2 = blankstack
for count = 1:imSequenceLength
  blankstack2[:,:,count] .= blankstack[:,:,imSequenceLength - count + 1]
end


first = ITVar.I_nextFrame[:,:]
for count = 1:imSequenceLength -1
  @show count
  ITVar.I_nextFrame[:,:] = blankstack[:,:,count+1]
  KTL_Tracker!(ITVar, ITConst)
  # @show ITVar.p_reference
  for columnCount = 1:length(ITVar.p_reference[1,:])
    if (ITVar.p_reference[2, columnCount][1]<1 || ITVar.p_reference[2, columnCount][2] <1)
    else
      draw!(drawimage, LineSegment(ITVar.p_reference[1, columnCount],ITVar.p_reference[2, columnCount]))
    end
  end
  # fillNewImageTemplates!(ITVar, ITConst)
  # updateFeatures!(ITVar, ITConst)
  # ImageView.imshow(ITVar.FixedMemory.I_warped)
  if (addBestShiTomasi!(ITVar, ITConst, halfWindowSize = 9))
    # fillNewImageTemplates!(ITVar, ITConst)
    @show updated = "updated+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
  end
end

@show reverse = "reverse"
# updateFeatures!(ITVar, ITConst)
fillNewImageTemplates!(ITVar, ITConst)
drawimage2 = Gray.(deepcopy(orgI))
second = ITVar.I_nextFrame[:,:]

for count = 1:imSequenceLength -1
    @show count
    ITVar.I_nextFrame[:,:] = blankstack2[:,:,count+1]
    KTL_Tracker!(ITVar, ITConst)
  #  @show ITVar.p_reference
   for columnCount = 1:length(ITVar.p_reference[1,:])
     if (ITVar.p_reference[2, columnCount][1]<1 || ITVar.p_reference[2, columnCount][2] <1)
     else
       draw!(drawimage2, LineSegment(ITVar.p_reference[1, columnCount],ITVar.p_reference[2, columnCount]))
     end
   end
   # fillNewImageTemplates!(ITVar, ITConst)
   # updateFeatures!(ITVar, ITConst)
  # ImageView.imshow(ITVar.FixedMemory.I_warped)
  if (addBestShiTomasi!(ITVar, ITConst, halfWindowSize = 9))
    # fillNewImageTemplates!(ITVar, ITConst)
    @show updated = "updated+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
  end
end


ImageView.imshow(drawimage)
ImageView.imshow(drawimage2)
third = ITVar.I_nextFrame[:,:]
# test = orgI .- blank
ImageView.imshow(blankstack)
ImageView.imshow(blankstack2)

# @test compare(corners[1].keypoint[1] + windowSize, ITVar.p_reference[2][1])
@test isapprox(corners[1].keypoint[1] + windowSize, ITVar.p_reference[2][1], atol = 8)
end
