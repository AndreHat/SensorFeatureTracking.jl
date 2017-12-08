using SensorFeatureTracking
using Base: Test


@testset "SensorFeatureTracking" begin
    # Common
    include("sensorgeometry.jl")
    include("blockflow.jl")
    include("KLTTrackingAlgorithmsTest.jl")
end
