#pragma once

#include "linear_algebra.h"

#define PI_OVER_TWO 1.5707963267948966192313216916397514420985

struct Camera
{
    Vec3 pos;
    Vec3 fd;
    Vec3 up;
    float focal_distance;
    float aperture_radius;
    int blades;
    float dummy0;
    
};

class InteractiveCamera
{
private:
    
    Vec3 centerPosition;
    Vec3 viewDirection;
    float yaw;
    float pitch;
    float apertureRadius;
    float focalDistance;
    
    void fixYaw();
    void fixPitch();
    void fixApertureRadius();
    void fixFocalDistance();
    
public:
    InteractiveCamera(Vec3 pos, Vec3 fd, Vec3 up, float focal_distance, float aperture_radius);
    virtual ~InteractiveCamera();
    void changeYaw(float m);
    void changePitch(float m);
    void changeRadius(float m);
    void changeAltitude(float m);
    void changeFocalDistance(float m);
    void strafe(float m);
    void goForward(float m);
    void rotateRight(float m);
    void changeApertureDiameter(float m);
    
    void buildRenderCamera(Camera* renderCamera);
};
