#include "camera.h"

// constructor and default values
InteractiveCamera::InteractiveCamera(Vec3 pos, Vec3 fd, Vec3 up, float focal_distance, float aperture_radius)
{
	centerPosition = pos;
    float fdy = fd.y;
    Vec3 woy = Vec3(fd.x, 0, fd.z);
    
	yaw = atan2(woy.x, woy.z);
    
	pitch = atan2(fdy, sqrt(woy.x*woy.x + woy.z*woy.z));
	apertureRadius = aperture_radius; // 0.04
	focalDistance = focal_distance;
}

InteractiveCamera::~InteractiveCamera() {}

void InteractiveCamera::changeYaw(float m){
	yaw += m;
	fixYaw();
}

void InteractiveCamera::changePitch(float m){
	pitch += m;
	fixPitch();
}

void InteractiveCamera::changeAltitude(float m){
	centerPosition.y += m;
	//fixCenterPosition();
}

void InteractiveCamera::goForward(float m){
	centerPosition += viewDirection * m;
}

void InteractiveCamera::strafe(float m){
	Vec3 strafeAxis = cross(viewDirection, Vec3(0, 1, 0));
	strafeAxis.normalize();
	centerPosition += strafeAxis * m;
}

void InteractiveCamera::rotateRight(float m){
	float yaw2 = yaw;
	yaw2 += m;
	float pitch2 = pitch;
	float xDirection = sin(yaw2) * cos(pitch2);
	float yDirection = sin(pitch2);
	float zDirection = cos(yaw2) * cos(pitch2);
	Vec3 directionToCamera = Vec3(xDirection, yDirection, zDirection);
	viewDirection = directionToCamera * (-1.0);
}

void InteractiveCamera::changeApertureDiameter(float m){
	apertureRadius += (apertureRadius + 0.01) * m; // Change proportional to current apertureRadius.
	fixApertureRadius();
}


void InteractiveCamera::changeFocalDistance(float m){
	focalDistance += m;
	fixFocalDistance();
}

float radiansToDegrees(float radians) {
	float degrees = radians * 180.0 / M_PI;
	return degrees;
}

float degreesToRadians(float degrees) {
	float radians = degrees / 180.0 * M_PI;
	return radians;
}

void InteractiveCamera::buildRenderCamera(Camera* renderCamera){
	float xDirection = sin(yaw) * cos(pitch);
	float yDirection = sin(pitch);
	float zDirection = cos(yaw) * cos(pitch);
	Vec3 directionToCamera = Vec3(xDirection, yDirection, zDirection);
	viewDirection = directionToCamera;
    Vec3 eyePosition = centerPosition;// + directionToCamera * radius;
	//Vec3 eyePosition = centerPosition; // rotate camera from stationary viewpoint

	renderCamera->pos = eyePosition;
	renderCamera->fd = viewDirection;
    Vec3 right = cross(Vec3(0, 1, 0), viewDirection);
    right.normalize();
    renderCamera->up = cross(viewDirection, right);
    renderCamera->up.normalize();
	renderCamera->aperture_radius = apertureRadius;
	renderCamera->focal_distance = focalDistance;
}

float mod(float x, float y) { // Does this account for -y ???
	return x - y * floorf(x / y);
}

void InteractiveCamera::fixYaw() {
	yaw = mod(yaw, 2 * M_PI); // Normalize the yaw.
}

float clamp2(float n, float low, float high) {
	n = fminf(n, high);
	n = fmaxf(n, low);
	return n;
}

void InteractiveCamera::fixPitch() {
	float padding = 0.05;
	pitch = clamp2(pitch, -PI_OVER_TWO + padding, PI_OVER_TWO - padding); // Limit the pitch.
}

void InteractiveCamera::fixApertureRadius() {
	float minApertureRadius = 0.0;
	float maxApertureRadius = 25.0;
	apertureRadius = clamp2(apertureRadius, minApertureRadius, maxApertureRadius);
}

void InteractiveCamera::fixFocalDistance() {
	float minFocalDist = 0.2;
	float maxFocalDist = 100.0;
	focalDistance = clamp2(focalDistance, minFocalDist, maxFocalDist);
}
