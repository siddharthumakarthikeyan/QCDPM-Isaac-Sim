// Sensor-based planning for the ground footprint of the team.
//
//   lidar_scan      2-D lidar model: ranges to obstacle footprints from a robot pose
//   OccupancyGrid   log-odds map built from scans taken at the robots' *estimated* poses, with a distance field
//   TeamPlanner     A* for the platform position with the formation centred on it: a cell is valid only if the four
//                   ground robots, the ground projections of their cables and the tool all clear the mapped
//                   obstacles; followed by line-of-sight shortcutting
//
// The map is two-dimensional, so an obstacle is treated as full height (the team never passes a cable over it), and
// unmapped space is treated as free: the plan has to be repeated as the map grows.
#pragma once

#include <vector>

#include "cdpr/types.hpp"

namespace cdpr {

struct Box2 {
  Vec2 lo, hi;
};

// n ranges over [-fov/2, fov/2] about the heading; r_max where nothing is hit.
VecX lidar_scan(const Vec3& pose, const std::vector<Box2>& boxes, int n, double fov, double r_min, double r_max);

class OccupancyGrid {
 public:
  OccupancyGrid(const Vec2& origin, double res, int nx, int ny);
  void insert_scan(const Vec3& pose, const VecX& ranges, double fov, double r_max);
  void update_distance();                     // call after inserting scans
  double distance(const Vec2& p) const;       // to the nearest occupied cell [m]; large outside the map
  bool occupied(int ix, int iy) const { return logodds[iy * nx + ix] > 0.7f; }
  bool known(int ix, int iy) const { return logodds[iy * nx + ix] != 0.0f; }
  Eigen::MatrixXf map() const;                // log-odds, rows = y
  Vec2 origin;
  double res;
  int nx, ny;

 private:
  void add(int ix, int iy, float v);
  std::vector<float> logodds, dist;
};

struct TeamFootprint {
  Mat42 ugv_offset = Mat42::Zero();  // ground-robot positions relative to the platform (formation centred)
  double robot_radius = 0.55, cable_margin = 0.2, body_radius = 0.35;
};

class TeamPlanner {
 public:
  TeamPlanner(const OccupancyGrid& g, const TeamFootprint& f) : grid(g), fp(f) {}
  bool valid(const Vec2& c) const;
  bool segment_valid(const Vec2& a, const Vec2& b) const;
  // Waypoints from start to goal (empty if none). `expanded` reports the search effort.
  std::vector<Vec2> plan(const Vec2& start, const Vec2& goal, int* expanded = nullptr) const;

 private:
  const OccupancyGrid& grid;
  TeamFootprint fp;
};

}  // namespace cdpr

// ---------------------------------------------------------------------------------------------------- 3-D
namespace cdpr {

struct Box3 {
  Vec3 lo, hi;
};

// Downward depth camera (n x n rays inside a square field of view about -z, yawed with the drone): the points
// where the rays hit the boxes or the ground, in the world frame. Rays longer than r_max return nothing.
MatX depth_scan(const Vec3& p, double yaw, const std::vector<Box3>& boxes, int n, double fov, double r_max);

// Height map: the highest point seen in each cell. Cells nobody has seen count as ground (height 0) until seen.
class ElevationGrid {
 public:
  ElevationGrid(const Vec2& origin, double res, int nx, int ny);
  void insert_points(const MatX& pts);                                   // rows x, y, z
  // A lidar return says only that something reaches the scan plane: raise the cell to at least that height.
  void insert_lidar(const Vec3& pose, const VecX& ranges, double fov, double r_max, double plane_height);
  // Rebuild the dilated maps used by the planner (max height within each radius). Call after inserting.
  void update(double robot_radius, double cable_margin, double body_radius);
  double height(const Vec2& p, int layer = 0) const;                     // layer 0 raw, 1 robot, 2 cable, 3 body
  Eigen::MatrixXf map() const;
  Eigen::MatrixXf seen() const;
  Vec2 origin;
  double res;
  int nx, ny;

 private:
  std::vector<float> h[4];
  std::vector<unsigned char> known;
};

struct TeamShape {
  Mat42 ugv_offset = Mat42::Zero();    // ground-robot fairleads relative to the platform centre (xy)
  Mat42 drone_offset = Mat42::Zero();  // drone fairleads relative to the platform centre (xy)
  Mat42 corner = Mat42::Zero();        // platform corners (xy), cable k attaches at corner k
  double fairlead_h = 0.35, drone_alt = 4.5, half_side = 0.15, tool_drop = 0.67;
  double robot_radius = 0.55, cable_margin = 0.2, body_radius = 0.35;
  double clearance = 0.15;             // vertical clearance of the cables and the tool above the map [m]
  double step_max = 0.10;              // the ground robots need ground no higher than this (above the depth noise:
                                       // the map keeps the highest return in a cell)
  double z_min = 0.9, z_max = 2.0, dz = 0.1;
  double climb_cost = 2.0;             // cost of a metre of height change relative to a metre of travel
  // Reconfiguration: the ground robots' ring may be scaled by one of `scales` (cable lengths change with it).
  // feasible(iz, is) != 0 marks the (height, scale) pairs the cables can hold (wrench-feasible); empty = all.
  std::vector<double> scales = {1.0};
  Eigen::MatrixXi feasible;
  double scale_cost = 3.0;             // cost of changing the scale by 1.0, in metres of travel
};

class TeamPlanner3 {
 public:
  TeamPlanner3(const ElevationGrid& g, const TeamShape& s) : grid(g), sh(s) {}
  bool valid(const Vec3& c) const;
  bool segment_valid(const Vec3& a, const Vec3& b) const;
  std::vector<Vec3> plan(const Vec3& start, const Vec3& goal, int* expanded = nullptr) const;
  // With reconfiguration: states (x, y, z, scale). The goal scale is free unless goal(3) > 0.
  bool valid4(const Vec4& c) const;
  bool segment_valid4(const Vec4& a, const Vec4& b) const;
  std::vector<Vec4> plan4(const Vec4& start, const Vec4& goal, int* expanded = nullptr) const;

 private:
  bool valid_scaled(const Vec3& c, double scale) const;
  const ElevationGrid& grid;
  TeamShape sh;
};

}  // namespace cdpr
