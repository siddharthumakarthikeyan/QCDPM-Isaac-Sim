#include "cdpr/planning.hpp"

#include <algorithm>
#include <queue>

namespace cdpr {

VecX lidar_scan(const Vec3& pose, const std::vector<Box2>& boxes, int n, double fov, double r_min, double r_max) {
  VecX r = VecX::Constant(n, r_max);
  const Vec2 o = pose.head<2>();
  for (int k = 0; k < n; ++k) {
    const double a = pose(2) - 0.5 * fov + fov * k / std::max(n - 1, 1);
    const Vec2 d(std::cos(a), std::sin(a));
    for (const Box2& b : boxes) {  // slab test
      double t0 = 0.0, t1 = r_max;
      bool hit = true;
      for (int ax = 0; ax < 2 && hit; ++ax) {
        if (std::abs(d(ax)) < 1e-12) {
          hit = o(ax) >= b.lo(ax) && o(ax) <= b.hi(ax);
        } else {
          double ta = (b.lo(ax) - o(ax)) / d(ax), tb = (b.hi(ax) - o(ax)) / d(ax);
          if (ta > tb) std::swap(ta, tb);
          t0 = std::max(t0, ta), t1 = std::min(t1, tb);
          hit = t0 <= t1;
        }
      }
      if (hit && t0 >= r_min) r(k) = std::min(r(k), t0);
    }
  }
  return r;
}

OccupancyGrid::OccupancyGrid(const Vec2& origin_, double res_, int nx_, int ny_)
    : origin(origin_), res(res_), nx(nx_), ny(ny_), logodds(nx_ * ny_, 0.0f), dist(nx_ * ny_, 1e6f) {}

void OccupancyGrid::add(int ix, int iy, float v) {
  if (ix < 0 || iy < 0 || ix >= nx || iy >= ny) return;
  float& l = logodds[iy * nx + ix];
  l = std::clamp(l + v, -4.0f, 4.0f);
}

void OccupancyGrid::insert_scan(const Vec3& pose, const VecX& ranges, double fov, double r_max) {
  const int n = static_cast<int>(ranges.size());
  for (int k = 0; k < n; ++k) {
    const double a = pose(2) - 0.5 * fov + fov * k / std::max(n - 1, 1);
    const bool hit = ranges(k) < r_max - 1e-6;
    const double len = ranges(k);
    // free cells along the beam (sampled at half a cell), then the end point
    const int steps = static_cast<int>(len / (0.5 * res));
    int lx = -1, ly = -1;
    for (int s = 0; s < steps; ++s) {
      const double t = s * 0.5 * res;
      const int ix = static_cast<int>(std::floor((pose(0) + t * std::cos(a) - origin(0)) / res));
      const int iy = static_cast<int>(std::floor((pose(1) + t * std::sin(a) - origin(1)) / res));
      if (ix == lx && iy == ly) continue;
      lx = ix, ly = iy;
      add(ix, iy, -0.2f);
    }
    if (hit)
      add(static_cast<int>(std::floor((pose(0) + len * std::cos(a) - origin(0)) / res)),
          static_cast<int>(std::floor((pose(1) + len * std::sin(a) - origin(1)) / res)), 0.9f);
  }
}

void OccupancyGrid::update_distance() {
  // two-pass chamfer distance transform (weights 1, sqrt 2), in metres
  const float a = static_cast<float>(res), b = static_cast<float>(res * std::sqrt(2.0));
  for (int i = 0; i < nx * ny; ++i) dist[i] = logodds[i] > 0.7f ? 0.0f : 1e6f;
  auto at = [&](int x, int y) -> float& { return dist[y * nx + x]; };
  for (int y = 0; y < ny; ++y)
    for (int x = 0; x < nx; ++x) {
      float& d = at(x, y);
      if (x > 0) d = std::min(d, at(x - 1, y) + a);
      if (y > 0) d = std::min(d, at(x, y - 1) + a);
      if (x > 0 && y > 0) d = std::min(d, at(x - 1, y - 1) + b);
      if (x < nx - 1 && y > 0) d = std::min(d, at(x + 1, y - 1) + b);
    }
  for (int y = ny - 1; y >= 0; --y)
    for (int x = nx - 1; x >= 0; --x) {
      float& d = at(x, y);
      if (x < nx - 1) d = std::min(d, at(x + 1, y) + a);
      if (y < ny - 1) d = std::min(d, at(x, y + 1) + a);
      if (x < nx - 1 && y < ny - 1) d = std::min(d, at(x + 1, y + 1) + b);
      if (x > 0 && y < ny - 1) d = std::min(d, at(x - 1, y + 1) + b);
    }
}

double OccupancyGrid::distance(const Vec2& p) const {
  const int ix = static_cast<int>(std::floor((p(0) - origin(0)) / res));
  const int iy = static_cast<int>(std::floor((p(1) - origin(1)) / res));
  if (ix < 0 || iy < 0 || ix >= nx || iy >= ny) return 1e6;
  return dist[iy * nx + ix];
}

Eigen::MatrixXf OccupancyGrid::map() const {
  return Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(logodds.data(), ny, nx);
}

bool TeamPlanner::valid(const Vec2& c) const {
  // The distance field is measured between cell centres (one cell of margin) and the chamfer metric overestimates
  // the Euclidean distance by up to 8%.
  const double q = grid.res, cf = 1.08;
  if (grid.distance(c) < cf * fp.body_radius + q) return false;
  for (int k = 0; k < NU; ++k) {
    const Vec2 o = fp.ugv_offset.row(k);
    if (grid.distance(c + o) < cf * fp.robot_radius + q) return false;
    const int n = std::max(2, static_cast<int>(o.norm() / grid.res));
    for (int s = 1; s < n; ++s)
      if (grid.distance(c + o * (static_cast<double>(s) / n)) < cf * fp.cable_margin + q) return false;
  }
  return true;
}

bool TeamPlanner::segment_valid(const Vec2& a, const Vec2& b) const {
  const int n = std::max(1, static_cast<int>((b - a).norm() / (0.5 * grid.res)));
  for (int s = 0; s <= n; ++s)
    if (!valid(a + (b - a) * (static_cast<double>(s) / n))) return false;
  return true;
}

std::vector<Vec2> TeamPlanner::plan(const Vec2& start, const Vec2& goal, int* expanded) const {
  const int nx = grid.nx, ny = grid.ny;
  auto cell = [&](const Vec2& p, int& ix, int& iy) {
    ix = std::clamp(static_cast<int>(std::floor((p(0) - grid.origin(0)) / grid.res)), 0, nx - 1);
    iy = std::clamp(static_cast<int>(std::floor((p(1) - grid.origin(1)) / grid.res)), 0, ny - 1);
  };
  auto centre = [&](int id) {
    return Vec2(grid.origin(0) + (id % nx + 0.5) * grid.res, grid.origin(1) + (id / nx + 0.5) * grid.res);
  };
  int sx, sy, gx, gy;
  cell(start, sx, sy);
  cell(goal, gx, gy);
  const int s_id = sy * nx + sx, g_id = gy * nx + gx;
  std::vector<float> g(nx * ny, 1e9f);
  std::vector<int> parent(nx * ny, -1);
  std::vector<signed char> ok(nx * ny, -1);  // validity cache
  auto is_ok = [&](int id) {
    if (ok[id] < 0) ok[id] = valid(centre(id)) ? 1 : 0;
    return ok[id] == 1;
  };
  auto h = [&](int id) {
    const int dx = std::abs(id % nx - gx), dy = std::abs(id / nx - gy);
    return static_cast<float>((std::max(dx, dy) + (std::sqrt(2.0) - 1.0) * std::min(dx, dy)));
  };
  using QE = std::pair<float, int>;
  std::priority_queue<QE, std::vector<QE>, std::greater<QE>> open;
  g[s_id] = 0.0f;
  open.push({h(s_id), s_id});
  int n_exp = 0;
  bool found = false;
  while (!open.empty()) {
    const auto [f, id] = open.top();
    open.pop();
    if (f > g[id] + h(id) + 1e-4f) continue;
    ++n_exp;
    if (id == g_id) {
      found = true;
      break;
    }
    const int x = id % nx, y = id / nx;
    for (int dy = -1; dy <= 1; ++dy)
      for (int dx = -1; dx <= 1; ++dx) {
        if (!dx && !dy) continue;
        const int x2 = x + dx, y2 = y + dy;
        if (x2 < 0 || y2 < 0 || x2 >= nx || y2 >= ny) continue;
        const int id2 = y2 * nx + x2;
        if (!is_ok(id2)) continue;
        const float c = g[id] + ((dx && dy) ? 1.41421356f : 1.0f);
        if (c < g[id2]) {
          g[id2] = c;
          parent[id2] = id;
          open.push({c + h(id2), id2});
        }
      }
  }
  if (expanded) *expanded = n_exp;
  std::vector<Vec2> path;
  if (!found) return path;
  for (int id = g_id; id != -1; id = parent[id]) path.push_back(centre(id));
  std::reverse(path.begin(), path.end());
  path.front() = start;
  path.back() = goal;
  // line-of-sight shortcutting
  std::vector<Vec2> out{path.front()};
  size_t i = 0;
  while (i + 1 < path.size()) {
    size_t j = path.size() - 1;
    while (j > i + 1 && !segment_valid(path[i], path[j])) --j;
    out.push_back(path[j]);
    i = j;
  }
  return out;
}

}  // namespace cdpr

// ---------------------------------------------------------------------------------------------------- 3-D
namespace cdpr {

MatX depth_scan(const Vec3& p, double yaw, const std::vector<Box3>& boxes, int n, double fov, double r_max) {
  MatX pts(n * n, 3);
  int m = 0;
  const double c = std::cos(yaw), s = std::sin(yaw), half = std::tan(0.5 * fov);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) {
      const double a = half * (2.0 * i / (n - 1) - 1.0), b = half * (2.0 * j / (n - 1) - 1.0);
      const Vec3 d = Vec3(c * a - s * b, s * a + c * b, -1.0).normalized();
      double t_hit = d.z() < 0 ? -p.z() / d.z() : r_max;  // the ground
      for (const Box3& bx : boxes) {
        double t0 = 0.0, t1 = r_max;
        bool hit = true;
        for (int ax = 0; ax < 3 && hit; ++ax) {
          if (std::abs(d(ax)) < 1e-12) {
            hit = p(ax) >= bx.lo(ax) && p(ax) <= bx.hi(ax);
          } else {
            double ta = (bx.lo(ax) - p(ax)) / d(ax), tb = (bx.hi(ax) - p(ax)) / d(ax);
            if (ta > tb) std::swap(ta, tb);
            t0 = std::max(t0, ta), t1 = std::min(t1, tb);
            hit = t0 <= t1;
          }
        }
        if (hit) t_hit = std::min(t_hit, t0);
      }
      if (t_hit < r_max) pts.row(m++) = (p + t_hit * d).transpose();
    }
  return pts.topRows(m);
}

ElevationGrid::ElevationGrid(const Vec2& origin_, double res_, int nx_, int ny_)
    : origin(origin_), res(res_), nx(nx_), ny(ny_), known(nx_ * ny_, 0) {
  for (auto& layer : h) layer.assign(nx * ny, 0.0f);
}

void ElevationGrid::insert_points(const MatX& pts) {
  for (int i = 0; i < pts.rows(); ++i) {
    const int ix = static_cast<int>(std::floor((pts(i, 0) - origin(0)) / res));
    const int iy = static_cast<int>(std::floor((pts(i, 1) - origin(1)) / res));
    if (ix < 0 || iy < 0 || ix >= nx || iy >= ny) continue;
    float& v = h[0][iy * nx + ix];
    v = std::max(v, static_cast<float>(pts(i, 2)));
    known[iy * nx + ix] = 1;
  }
}

void ElevationGrid::insert_lidar(const Vec3& pose, const VecX& ranges, double fov, double r_max,
                                 double plane_height) {
  const int n = static_cast<int>(ranges.size());
  for (int k = 0; k < n; ++k) {
    if (ranges(k) >= r_max - 1e-6) continue;
    const double a = pose(2) - 0.5 * fov + fov * k / std::max(n - 1, 1);
    const int ix = static_cast<int>(std::floor((pose(0) + ranges(k) * std::cos(a) - origin(0)) / res));
    const int iy = static_cast<int>(std::floor((pose(1) + ranges(k) * std::sin(a) - origin(1)) / res));
    if (ix < 0 || iy < 0 || ix >= nx || iy >= ny) continue;
    float& v = h[0][iy * nx + ix];
    v = std::max(v, static_cast<float>(plane_height));
  }
}

void ElevationGrid::update(double robot_radius, double cable_margin, double body_radius) {
  const double radii[3] = {robot_radius, cable_margin, body_radius};
  std::vector<float> tmp(nx * ny);
  for (int l = 0; l < 3; ++l) {
    // square dilation (max filter), separable; one extra cell for the quantisation. Conservative for a disc.
    const int w = static_cast<int>(std::ceil(radii[l] / res)) + 1;
    for (int y = 0; y < ny; ++y)
      for (int x = 0; x < nx; ++x) {
        float m = 0.0f;
        for (int k = std::max(0, x - w); k <= std::min(nx - 1, x + w); ++k) m = std::max(m, h[0][y * nx + k]);
        tmp[y * nx + x] = m;
      }
    for (int y = 0; y < ny; ++y)
      for (int x = 0; x < nx; ++x) {
        float m = 0.0f;
        for (int k = std::max(0, y - w); k <= std::min(ny - 1, y + w); ++k) m = std::max(m, tmp[k * nx + x]);
        h[l + 1][y * nx + x] = m;
      }
  }
}

double ElevationGrid::height(const Vec2& p, int layer) const {
  const int ix = static_cast<int>(std::floor((p(0) - origin(0)) / res));
  const int iy = static_cast<int>(std::floor((p(1) - origin(1)) / res));
  if (ix < 0 || iy < 0 || ix >= nx || iy >= ny) return 0.0;
  return h[layer][iy * nx + ix];
}

Eigen::MatrixXf ElevationGrid::map() const {
  return Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(h[0].data(), ny, nx);
}

Eigen::MatrixXf ElevationGrid::seen() const {
  Eigen::MatrixXf m(ny, nx);
  for (int y = 0; y < ny; ++y)
    for (int x = 0; x < nx; ++x) m(y, x) = known[y * nx + x];
  return m;
}

bool TeamPlanner3::valid(const Vec3& c) const { return valid_scaled(c, 1.0); }

bool TeamPlanner3::valid4(const Vec4& c) const {
  if (sh.feasible.size() > 0) {  // the cables must be able to hold the platform at this height and scale
    const int iz = std::clamp(static_cast<int>(std::round((c(2) - sh.z_min) / sh.dz)), 0,
                              static_cast<int>(sh.feasible.rows()) - 1);
    int is = 0;
    for (size_t k = 1; k < sh.scales.size(); ++k)
      if (std::abs(sh.scales[k] - c(3)) < std::abs(sh.scales[is] - c(3))) is = static_cast<int>(k);
    if (!sh.feasible(iz, is)) return false;
  }
  return valid_scaled(c.head<3>(), c(3));
}

bool TeamPlanner3::segment_valid4(const Vec4& a, const Vec4& b) const {
  const double len = std::max((b - a).head<3>().norm(), std::abs(b(3) - a(3)) * 2.75);
  const int n = std::max(1, static_cast<int>(len / (0.5 * grid.res)));
  for (int s = 0; s <= n; ++s) {
    Vec4 q = a + (b - a) * (static_cast<double>(s) / n);
    if (!valid_scaled(q.head<3>(), q(3))) return false;
  }
  return valid4(a) && valid4(b);
}

std::vector<Vec4> TeamPlanner3::plan4(const Vec4& start, const Vec4& goal, int* expanded) const {
  const int nx = grid.nx, ny = grid.ny, ns = static_cast<int>(sh.scales.size());
  const int nz = static_cast<int>(std::round((sh.z_max - sh.z_min) / sh.dz)) + 1;
  const size_t layer = static_cast<size_t>(nx) * ny * nz;
  auto centre = [&](size_t id) {
    const size_t sp = id % layer;
    const int x = sp % nx, y = (sp / nx) % ny, z = static_cast<int>(sp / (static_cast<size_t>(nx) * ny));
    Vec4 c;
    c << grid.origin(0) + (x + 0.5) * grid.res, grid.origin(1) + (y + 0.5) * grid.res, sh.z_min + z * sh.dz,
        sh.scales[id / layer];
    return c;
  };
  auto cell = [&](const Vec4& p) {
    const int x = std::clamp(static_cast<int>(std::floor((p(0) - grid.origin(0)) / grid.res)), 0, nx - 1);
    const int y = std::clamp(static_cast<int>(std::floor((p(1) - grid.origin(1)) / grid.res)), 0, ny - 1);
    const int z = std::clamp(static_cast<int>(std::round((p(2) - sh.z_min) / sh.dz)), 0, nz - 1);
    int is = 0;
    for (int k = 1; k < ns; ++k)
      if (std::abs(sh.scales[k] - p(3)) < std::abs(sh.scales[is] - p(3))) is = k;
    return static_cast<size_t>(is) * layer + (static_cast<size_t>(z) * ny + y) * nx + x;
  };
  const size_t s_id = cell(start);
  const Vec4 sc = centre(s_id);
  const bool free_goal_scale = goal(3) <= 0.0;
  const size_t g_sp = cell(Vec4(goal(0), goal(1), goal(2), sh.scales[0])) % layer;
  const Vec4 gc = centre(g_sp);
  const size_t N = layer * ns;
  std::vector<float> g(N, 1e9f);
  std::vector<int> parent(N, -1);
  std::vector<signed char> ok(N, -1);
  auto is_ok = [&](size_t id) {
    if (ok[id] < 0) {
      const Vec4 p = centre(id);
      const bool near_start = (p - sc).head<2>().norm() <= 0.3 && std::abs(p(2) - sc(2)) <= 0.2 && p(3) == sc(3);
      ok[id] = (valid4(p) || near_start) ? 1 : 0;
    }
    return ok[id] == 1;
  };
  auto h = [&](size_t id) {
    const Vec4 p = centre(id);
    return static_cast<float>((p.head<2>() - gc.head<2>()).norm() + sh.climb_cost * std::abs(p(2) - gc(2)) +
                              (free_goal_scale ? 0.0 : sh.scale_cost * std::abs(p(3) - goal(3))));
  };
  using QE = std::pair<float, size_t>;
  std::priority_queue<QE, std::vector<QE>, std::greater<QE>> open;
  g[s_id] = 0.0f;
  open.push({h(s_id), s_id});
  int n_exp = 0;
  long found = -1;
  while (!open.empty()) {
    const auto [f, id] = open.top();
    open.pop();
    if (f > g[id] + h(id) + 1e-4f) continue;
    ++n_exp;
    if (id % layer == g_sp && (free_goal_scale || std::abs(sh.scales[id / layer] - goal(3)) < 1e-9)) {
      found = static_cast<long>(id);
      break;
    }
    const size_t sp = id % layer;
    const int x = sp % nx, y = (sp / nx) % ny, z = static_cast<int>(sp / (static_cast<size_t>(nx) * ny));
    const int is = static_cast<int>(id / layer);
    auto relax = [&](size_t id2, float step) {
      if (!is_ok(id2)) return;
      const float c = g[id] + step;
      if (c < g[id2]) {
        g[id2] = c;
        parent[id2] = static_cast<int>(id);
        open.push({c + h(id2), id2});
      }
    };
    for (int dz = -1; dz <= 1; ++dz)
      for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx) {
          if (!dx && !dy && !dz) continue;
          const int x2 = x + dx, y2 = y + dy, z2 = z + dz;
          if (x2 < 0 || y2 < 0 || z2 < 0 || x2 >= nx || y2 >= ny || z2 >= nz) continue;
          relax(static_cast<size_t>(is) * layer + (static_cast<size_t>(z2) * ny + y2) * nx + x2,
                static_cast<float>(grid.res * std::sqrt(static_cast<double>(dx * dx + dy * dy)) +
                                   sh.climb_cost * sh.dz * std::abs(dz)));
        }
    for (int ds = -1; ds <= 1; ds += 2) {  // reconfigure in place
      const int is2 = is + ds;
      if (is2 < 0 || is2 >= ns) continue;
      relax(static_cast<size_t>(is2) * layer + sp,
            static_cast<float>(sh.scale_cost * std::abs(sh.scales[is2] - sh.scales[is])));
    }
  }
  if (expanded) *expanded = n_exp;
  std::vector<Vec4> path;
  if (found < 0) return path;
  for (long id = found; id != -1; id = parent[id]) path.push_back(centre(static_cast<size_t>(id)));
  std::reverse(path.begin(), path.end());
  path.front() = start;
  path.back().head<3>() = goal.head<3>();
  std::vector<Vec4> out{path.front()};
  size_t i = 0;
  while (i + 1 < path.size()) {
    size_t j = path.size() - 1;
    while (j > i + 1 && !segment_valid4(path[i], path[j])) --j;
    out.push_back(path[j]);
    i = j;
  }
  return out;
}

bool TeamPlanner3::valid_scaled(const Vec3& c, double scale) const {
  const Vec2 xy = c.head<2>();
  if (c.z() < sh.z_min - 1e-9 || c.z() > sh.z_max + 1e-9) return false;
  // tool tip above whatever is under the platform
  if (grid.height(xy, 3) + sh.clearance > c.z() - sh.tool_drop) return false;
  for (int k = 0; k < NU; ++k) {
    // ground robot on flat ground
    const Vec2 g = xy + scale * sh.ugv_offset.row(k).transpose();
    if (grid.height(g, 1) > sh.step_max) return false;
    // lower cable: bottom corner -> ground fairlead; upper cable: top corner -> drone fairlead
    const Vec2 a = xy + sh.corner.row(k).transpose();
    const Vec2 d = xy + sh.drone_offset.row(k).transpose();
    const int n = std::max(2, static_cast<int>((g - a).norm() / grid.res));
    for (int s = 1; s < n; ++s) {
      const double u = static_cast<double>(s) / n;
      const double z = (c.z() - sh.half_side) * (1.0 - u) + sh.fairlead_h * u;
      if (grid.height(a + u * (g - a), 2) + sh.clearance > z && grid.height(a + u * (g - a), 2) > sh.step_max)
        return false;
    }
    const int m = std::max(2, static_cast<int>((d - a).norm() / grid.res));
    for (int s = 0; s <= m; ++s) {
      const double u = static_cast<double>(s) / m;
      const double z = (c.z() + sh.half_side) * (1.0 - u) + sh.drone_alt * u;
      if (grid.height(a + u * (d - a), 2) + sh.clearance > z) return false;
    }
  }
  return true;
}

bool TeamPlanner3::segment_valid(const Vec3& a, const Vec3& b) const {
  const int n = std::max(1, static_cast<int>((b - a).norm() / (0.5 * grid.res)));
  for (int s = 0; s <= n; ++s)
    if (!valid(a + (b - a) * (static_cast<double>(s) / n))) return false;
  return true;
}

std::vector<Vec3> TeamPlanner3::plan(const Vec3& start, const Vec3& goal, int* expanded) const {
  const int nx = grid.nx, ny = grid.ny;
  const int nz = static_cast<int>(std::round((sh.z_max - sh.z_min) / sh.dz)) + 1;
  auto idx = [&](int x, int y, int z) { return (z * ny + y) * nx + x; };
  auto centre = [&](int id) {
    const int x = id % nx, y = (id / nx) % ny, z = id / (nx * ny);
    return Vec3(grid.origin(0) + (x + 0.5) * grid.res, grid.origin(1) + (y + 0.5) * grid.res, sh.z_min + z * sh.dz);
  };
  auto cell = [&](const Vec3& p) {
    const int x = std::clamp(static_cast<int>(std::floor((p(0) - grid.origin(0)) / grid.res)), 0, nx - 1);
    const int y = std::clamp(static_cast<int>(std::floor((p(1) - grid.origin(1)) / grid.res)), 0, ny - 1);
    const int z = std::clamp(static_cast<int>(std::round((p(2) - sh.z_min) / sh.dz)), 0, nz - 1);
    return idx(x, y, z);
  };
  const int s_id = cell(start), g_id = cell(goal);
  const Vec3 gc = centre(g_id);
  const size_t N = static_cast<size_t>(nx) * ny * nz;
  std::vector<float> g(N, 1e9f);
  std::vector<int> parent(N, -1);
  std::vector<signed char> ok(N, -1);
  // New measurements can put the cell the team is in just outside the valid set (the map keeps the highest noisy
  // return). Cells within 0.3 m of the start are therefore passable, so the search can leave instead of failing.
  const Vec3 sc = centre(s_id);
  auto is_ok = [&](int id) {
    if (ok[id] < 0) {
      const Vec3 p = centre(id);
      ok[id] = (valid(p) || ((p - sc).head<2>().norm() <= 0.3 && std::abs(p.z() - sc.z()) <= 0.2)) ? 1 : 0;
    }
    return ok[id] == 1;
  };
  // metric: horizontal distance + climb_cost x height change (admissible heuristic in the same metric)
  auto h = [&](int id) {
    const Vec3 p = centre(id);
    return static_cast<float>((p.head<2>() - gc.head<2>()).norm() + sh.climb_cost * std::abs(p.z() - gc.z()));
  };
  using QE = std::pair<float, int>;
  std::priority_queue<QE, std::vector<QE>, std::greater<QE>> open;
  g[s_id] = 0.0f;
  open.push({h(s_id), s_id});
  int n_exp = 0;
  bool found = false;
  while (!open.empty()) {
    const auto [f, id] = open.top();
    open.pop();
    if (f > g[id] + h(id) + 1e-4f) continue;
    ++n_exp;
    if (id == g_id) {
      found = true;
      break;
    }
    const int x = id % nx, y = (id / nx) % ny, z = id / (nx * ny);
    for (int dz = -1; dz <= 1; ++dz)
      for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx) {
          if (!dx && !dy && !dz) continue;
          const int x2 = x + dx, y2 = y + dy, z2 = z + dz;
          if (x2 < 0 || y2 < 0 || z2 < 0 || x2 >= nx || y2 >= ny || z2 >= nz) continue;
          const int id2 = idx(x2, y2, z2);
          if (!is_ok(id2)) continue;
          const float c = g[id] + static_cast<float>(grid.res * std::sqrt(static_cast<double>(dx * dx + dy * dy)) +
                                                     sh.climb_cost * sh.dz * std::abs(dz));
          if (c < g[id2]) {
            g[id2] = c;
            parent[id2] = id;
            open.push({c + h(id2), id2});
          }
        }
  }
  if (expanded) *expanded = n_exp;
  std::vector<Vec3> path;
  if (!found) return path;
  for (int id = g_id; id != -1; id = parent[id]) path.push_back(centre(id));
  std::reverse(path.begin(), path.end());
  path.front() = start;
  path.back() = goal;
  std::vector<Vec3> out{path.front()};
  size_t i = 0;
  while (i + 1 < path.size()) {
    size_t j = path.size() - 1;
    while (j > i + 1 && !segment_valid(path[i], path[j])) --j;
    out.push_back(path[j]);
    i = j;
  }
  return out;
}

}  // namespace cdpr
