import numpy as np
import cv2
import os
import time

def RT_to_rvec_tvec(RT):
	rvec = cv2.Rodrigues(RT[:3,:3])[0]
	tvec = RT[:3,3]
	return rvec, tvec

def current_milli_time():
	return round(time.time() * 1000)

def img_float_to_uint8(img_float):
	img_uint8 = np.clip(np.around(img_float).astype(np.int),0,255).astype(np.uint8)
	return img_uint8

class MyObject():
	def __init__(self):
		self.t = ''
		self.set_phong_material()
	def set_ambient_light(self, illumination):
		self.t = 'ambient_light'
		self.i = float(illumination)
	def set_point_light(self, x, y, z, i_diffused, i_specular):
		self.t = 'point_light'
		self.x = float(x)
		self.y = float(y)
		self.z = float(z)
		self.i_diffused = float(i_diffused)
		self.i_specular = float(i_specular)
	def set_cylinder_light(self, x_start, y_start, z_start, x_u, y_u, z_u, r, i_diffused, i_specular):
		# specify a ray
		self.t = 'cylinder_light'
		self.light_start = np.array([x_start, y_start, z_start], dtype=np.float64)
		self.light_unit_vec = np.array([x_u, y_u, z_u], dtype=np.float64)
		assert np.linalg.norm(self.light_unit_vec) > 0
		self.light_unit_vec /= np.linalg.norm(self.light_unit_vec)
		self.r = float(r)
		self.i_diffused = float(i_diffused)
		self.i_specular = float(i_specular)
	def set_cone_light(self, x_start, y_start, z_start, x_u, y_u, z_u, angle, i_diffused, i_specular):
		# specify a ray
		self.t = 'cone_light'
		self.light_start = np.array([x_start, y_start, z_start], dtype=np.float64)
		self.light_unit_vec = np.array([x_u, y_u, z_u], dtype=np.float64)
		assert np.linalg.norm(self.light_unit_vec) > 0
		self.light_unit_vec /= np.linalg.norm(self.light_unit_vec)
		assert angle > 0
		if angle > 360:
			self.min_inner = 0
		else:
			self.min_inner = np.cos(float(np.deg2rad(angle))*0.5)
		self.i_diffused = float(i_diffused)
		self.i_specular = float(i_specular)
	def set_sphere(self, x, y, z, r):
		self.t = 'sphere'
		self.x = float(x)
		self.y = float(y)
		self.z = float(z)
		self.r = float(r)
	def set_plane(self, x_coeff, y_coeff, z_coeff, xyz_sum):
		self.t = 'plane'
		len_xyz_sq = x_coeff*x_coeff + y_coeff*y_coeff + z_coeff*z_coeff
		assert len_xyz_sq > 0
		len_xyz = np.sqrt(len_xyz_sq)
		self.x = float(x_coeff) / len_xyz
		self.y = float(y_coeff) / len_xyz
		self.z = float(z_coeff) / len_xyz
		self.xyz_sum = float(xyz_sum/len_xyz)
	def set_triangle(self, x1, y1, z1, x2, y2, z2, x3, y3, z3):
		self.t = 'triangle'
		self.x1 = float(x1)
		self.y1 = float(y1)
		self.z1 = float(z1)
		self.x2 = float(x2)
		self.y2 = float(y2)
		self.z2 = float(z2)
		self.x3 = float(x3)
		self.y3 = float(y3)
		self.z3 = float(z3)
	def set_phong_material(self, ks=0, kd=0, ka=1, alpha=1):
		self.ks = float(ks) # specular reflection
		self.kd = float(kd) # diffuse reflection constant
		self.ka = float(ka) # ambient reflection constant
		self.alpha = float(alpha) # shininess
	def __str__(self):
		ret = ''
		m_str = '{ks=%f, kd=%f, ka=%f, alpha=%f}' % (self.ks, self.kd, self.ka, self.alpha)
		if self.t == 'ambient_light':
			ret += '{%s %f}' % (self.t, self.i)
		elif self.t == 'point_light':
			ret += '{%s (%f,%f,%f) i_diffused=%f i_specular=%f}' % (self.t, self.x, self.y, self.z, self.i_diffused, self.i_specular)
		elif self.t == 'cylinder_light':
			ret += '{%s light_start=%s, light_unit_vec=%s, r=%f, i_diffused=%f i_specular=%f}' % (self.t, self.light_start, self.light_unit_vec, self.r, self.i_diffused, self.i_specular)
		elif self.t == 'cone_light':
			ret += '{%s light_start=%s, light_unit_vec=%s, min_inner=%f, i_diffused=%f i_specular=%f}' % (self.t, self.light_start, self.light_unit_vec, self.min_inner, self.i_diffused, self.i_specular)
		elif self.t == 'sphere':
			ret += '{%s (%f,%f,%f) %f %s}' % (self.t, self.x, self.y, self.z, self.r, m_str)
		elif self.t == 'plane':
			ret += '{%s (%f,%f,%f) %f %s}' % (self.t, self.x, self.y, self.z, self.xyz_sum, m_str)
		elif self.t == 'triangle':
			ret += '{%s (%f,%f,%f),(%f,%f,%f),(%f,%f,%f) %s}' % (self.t,self.x1,self.y1,self.z1,self.x2,self.y2,self.z2,self.x3,self.y3,self.z3,m_str)
		return ret

class MyScene():
	def __init__(self):
		self.object_list = []
		self.set_background_color()
	def set_background_color(self, background_color=0):
		self.background_color = float(background_color)
	def add_object(self, obj):
		self.object_list.append(obj)
	def primary_ray_tracing(self, pt_start, unit_vec):
		# assume no inter-reflection
		first_obj = self.internal_get_first_object(pt_start, unit_vec)
		if first_obj is None:
			return self.background_color
		# return illumination
		Ip = 0
		# find all ambient light sum
		ia = self.internal_get_ambient_light_sum()
		Ip += ia * first_obj.ka
		if first_obj.t == 'sphere':
			first_xyz = np.array([first_obj.x, first_obj.y, first_obj.z], dtype=np.float64)
			V = pt_start - first_xyz
			V /= np.linalg.norm(V)
			_, k = self.internal_intersect_sphere(first_xyz, first_obj.r, pt_start, unit_vec)
			intersection = pt_start + k * unit_vec
			N = intersection - first_xyz
			N /= np.linalg.norm(N)
		elif first_obj.t == 'plane':
			_, k = self.internal_intersect_plane(first_obj.x, first_obj.y, first_obj.z, first_obj.xyz_sum, pt_start, unit_vec)
			intersection = pt_start + k * unit_vec
			V = (-1)* unit_vec
			N = np.array([first_obj.x, first_obj.y, first_obj.z], dtype=np.float64)
			N /= np.linalg.norm(N)
		elif first_obj.t == 'triangle':
			_, k = self.internal_intersect_triangle(first_obj.x1,first_obj.y1,first_obj.z1,first_obj.x2,first_obj.y2,first_obj.z2,first_obj.x3,first_obj.y3,first_obj.z3,pt_start,unit_vec)
			intersection = pt_start + k * unit_vec
			V = (-1)* unit_vec
			N = self.internal_triangle_normal(first_obj.x1,first_obj.y1,first_obj.z1,first_obj.x2,first_obj.y2,first_obj.z2,first_obj.x3,first_obj.y3,first_obj.z3)
			N /= np.linalg.norm(N)

		# calculate light
		for obj in self.object_list:
			if obj.t == 'point_light':
				light_xyz = np.array([obj.x, obj.y, obj.z], dtype=np.float64)
			elif obj.t == 'cylinder_light':
				light_xyz = obj.light_start
			elif obj.t == 'cone_light':
				light_xyz = obj.light_start
			else:
				continue # not light source

			unit_vec = intersection - light_xyz
			unit_vec = unit_vec / np.linalg.norm(unit_vec)
			light_see_first_obj = self.internal_get_first_object(light_xyz, unit_vec)
			if light_see_first_obj != first_obj:
				continue

			if obj.t == 'cylinder_light':
				# TODO: check whether in cylinder
				# obj.light_start
				k = np.inner(obj.light_unit_vec, intersection-obj.light_start)
				if k < 0:
					continue
				d = (obj.light_start + k*obj.light_unit_vec - intersection)
				if np.linalg.norm(d) > obj.r:
					# print(np.linalg.norm(d), obj.r)
					# exit()
					continue
			if obj.t == 'cone_light':
				# TODO: check whether in cone
				v1 = intersection-obj.light_start
				v1 /= np.linalg.norm(v1)
				if np.inner(v1, obj.light_unit_vec) < obj.min_inner:
					continue

			can_see_light = True
			if first_obj.t == 'sphere':
				# if the same object, check whether the same point
				light_to_sphere_surface = intersection - light_xyz
				light_to_sphere_center = first_xyz - light_xyz
				if np.inner(light_to_sphere_surface,light_to_sphere_surface) > \
					np.inner(light_to_sphere_center, light_to_sphere_surface):
					can_see_light = False		
						
			if can_see_light:
				# if first_obj.t in ['triangle']:
					# print('triangle_Ip', Ip)
				Lm = light_xyz - intersection
				Lm /= np.linalg.norm(Lm)
				Rm = 2*np.inner(Lm,N)*N - Lm
				imd = obj.i_diffused
				ims = obj.i_specular
				if first_obj.t in ['plane', 'triangle']:
					Ip += first_obj.kd*np.abs(np.inner(Lm,N))*imd # diffused
					# print('np.inner(Lm,N)',np.inner(Lm,N))
					# print('np.abs(np.inner(Lm,N))',np.abs(np.inner(Lm,N)))
					# print('first_obj.kd*np.abs(np.inner(Lm,N))*imd', first_obj.kd*np.abs(np.inner(Lm,N))*imd)
					# print('triangle_Ip_gj', Ip)
					Ip += first_obj.ks*np.power(np.abs(np.inner(Rm,V)),first_obj.alpha)*ims # specular
					# print('triangle_Ip_end', Ip)
					# print('kd', first_obj.kd)
					# print('ks', first_obj.ks)
					# print('Lm', Lm)
					# print('N', N)
					# print('imd', imd)
					# print('Rm', Rm)
					# print('V', V)
					# print('alpha', first_obj.alpha)
					# print('ims', ims)
					# exit()
				else:
					Ip += first_obj.kd*np.inner(Lm,N)*imd # diffused
					Ip += first_obj.ks*np.power(np.abs(np.inner(Rm,V)),first_obj.alpha)*ims # specular
				# 	exit()

		return Ip

	def internal_get_ambient_light_sum(self):
		ia = 0
		for obj in self.object_list:
			if obj.t == 'ambient_light':
				ia += obj.i
		return ia

	def internal_get_first_object(self, pt_start, unit_vec):
		tmp_obj = None
		tmp_min_k = np.Inf
		for obj in self.object_list:
			if obj.t == 'sphere':
				xyz = np.array([obj.x, obj.y, obj.z], dtype=np.float64)
				ret, k = self.internal_intersect_sphere(xyz, obj.r, pt_start, unit_vec)
				if ret:
					if k < tmp_min_k:
						tmp_obj = obj
						tmp_min_k = k
			elif obj.t == 'plane':
				ret, k = self.internal_intersect_plane(obj.x, obj.y, obj.z, obj.xyz_sum, pt_start, unit_vec)
				if ret:
					if k < tmp_min_k:
						tmp_obj = obj
						tmp_min_k = k
			elif obj.t == 'triangle':
				ret, k = self.internal_intersect_triangle(obj.x1,obj.y1,obj.z1,obj.x2,obj.y2,obj.z2,obj.x3,obj.y3,obj.z3,pt_start,unit_vec)
				if ret:
					if k < tmp_min_k:
						tmp_obj = obj
						tmp_min_k = k
		# exit()
		return tmp_obj

	def internal_three_point_order(self, p1, p2, p3):
		x1,y1 = p1[0], p1[1]
		x2,y2 = p2[0], p2[1]
		x3,y3 = p3[0], p3[1]
		return np.sign(x1*y2+x2*y3+x3*y1-x1*y3-x2*y1-x3*y2)

	def internal_triangle_normal(self, x1, y1, z1, x2, y2, z2, x3, y3, z3):
		p1 = np.array([x1,y1,z1], dtype=np.float64)
		p2 = np.array([x2,y2,z2], dtype=np.float64)
		p3 = np.array([x3,y3,z3], dtype=np.float64)
		v12 = p2 - p1
		v13 = p3 - p1
		v23 = p3 - p2
		assert np.linalg.norm(v12) > 0
		assert np.linalg.norm(v13) > 0
		assert np.linalg.norm(v23) > 0
		uv12 = v12 / np.linalg.norm(v12)
		uv13 = v13 / np.linalg.norm(v13)
		uv23 = v23 / np.linalg.norm(v23)
		assert abs(np.inner(uv12,uv13)) < 1
		assert abs(np.inner(uv12,uv23)) < 1
		assert abs(np.inner(uv13,uv23)) < 1
		uvs = [abs(np.inner(uv12,uv13)), abs(np.inner(uv12,uv23)), abs(np.inner(uv13,uv23))]
		# use the minumum one to calculate normal vector for numerically stablility
		idx = np.argmin(uvs)
		if idx == 0:
			tmp_normal = np.cross(uv12,uv13)
		elif idx == 1:
			tmp_normal = np.cross(uv12,uv23)
		else:
			tmp_normal = np.cross(uv13,uv23)
		# print('triangle_normal', tmp_normal)
		return tmp_normal

	def internal_intersect_triangle(self, x1, y1, z1, x2, y2, z2, x3, y3, z3, pt_start, unit_vec):
		tmp_normal = self.internal_triangle_normal(x1, y1, z1, x2, y2, z2, x3, y3, z3)
		p1 = np.array([x1,y1,z1], dtype=np.float64)
		p2 = np.array([x2,y2,z2], dtype=np.float64)
		p3 = np.array([x3,y3,z3], dtype=np.float64)
		v12 = p2 - p1
		uv12 = v12 / np.linalg.norm(v12)
		xyz_sum = np.inner(tmp_normal, p1)
		# print(np.inner(tmp_normal,p1),np.inner(tmp_normal,p2),np.inner(tmp_normal,p3))
		# print(tmp_normal)
		ret, k = self.internal_intersect_plane(tmp_normal[0], tmp_normal[1], tmp_normal[2], xyz_sum, pt_start, unit_vec)
		# print(ret, k)
		if not ret:
			return False, 0.

		pd = pt_start + k*unit_vec
		third_axis = np.cross(uv12, tmp_normal)
		# use uv12, third_axis, tmp_normal to represent p1, p2, p3, pd
		p1_t = np.array([np.inner(p1, uv12), np.inner(p1, third_axis), np.inner(p1, tmp_normal)], dtype=np.float64)
		p2_t = np.array([np.inner(p2, uv12), np.inner(p2, third_axis), np.inner(p2, tmp_normal)], dtype=np.float64)
		p3_t = np.array([np.inner(p3, uv12), np.inner(p3, third_axis), np.inner(p3, tmp_normal)], dtype=np.float64)
		pd_t = np.array([np.inner(pd, uv12), np.inner(pd, third_axis), np.inner(pd, tmp_normal)], dtype=np.float64)
		# print(p1_t, p2_t, p3_t, pd_t)
		# check whether inside the triangle
		o123 = self.internal_three_point_order(p1_t,p2_t,p3_t)
		o12d = self.internal_three_point_order(p1_t,p2_t,pd_t)
		o23d = self.internal_three_point_order(p2_t,p3_t,pd_t)
		o31d = self.internal_three_point_order(p3_t,p1_t,pd_t)
		if o123*o12d > 0 and o123*o23d > 0 and o123*o31d > 0:
			return True, k
		return False, 0.


	def internal_intersect_plane(self, x_coeff, y_coeff, z_coeff, xyz_sum, pt_start, unit_vec):
		# x_coeff * x + y_coeff * y + z_coeff * z = xyz_sum
		plane_normal = np.array([x_coeff,y_coeff,z_coeff], dtype=np.float64)
		if np.inner(plane_normal,unit_vec) == 0:
			# check whether on plane or not,
			# if not on plane, no intersection,
			# if on plane, return pt_start
			if np.inner(pt_start,plane_normal) == xyz_sum:
				return True, 0.
			else:
				return False, 0.
		# np.inner(plane_normal, pt_start + k*unit_vec) = xyz_sum, so
		# np.inner(plane_normal,pt_start) + k * np.inner(plane_normal,unit_vec) = xyz_sum, so
		# k = (xyz_sum - np.inner(plane_normal,pt_start)) / np.inner(plane_normal,unit_vec)
		k = (xyz_sum - np.inner(plane_normal,pt_start)) / np.inner(plane_normal,unit_vec)
		# print(plane_normal, xyz_sum)
		# print(pt_start + k*unit_vec)
		# print('gj')
		# exit()
		return True, k


	def internal_intersect_sphere(self, sphere_center, sphere_radius, pt_start, unit_vec):
		A = pt_start
		B = unit_vec
		P = sphere_center
		r = sphere_radius
		k = np.inner(B, P-A)
		d = (A + k*B - P)
		d_norm = np.linalg.norm(d)
		if d_norm > r:
			return False, 0.
		k_delta = np.sqrt((r*r-d_norm*d_norm))
		intersections_k = [k-k_delta, k+k_delta]
		intersections_k = [x for x in intersections_k if x > 0]
		if len(intersections_k) == 0:
			return False, 0.
		good_k = intersections_k[0] # smallest
		return True, good_k

	def __str__(self):
		ret = ''
		for obj in self.object_list:
			ret += '%s\n' % obj
		return ret


class MyRenderer():
	def __init__(self, K, RT, dist, width, height):
		self.K = K.copy()
		self.RT = RT.copy()
		self.dist = dist.copy()
		self.width = int(width)
		self.height = int(height)
		assert self.width > 0
		assert self.height > 0
	def render_primary_ray_tracing(self, scene):
		print(scene)
		img_float = np.zeros([self.height, self.width], dtype=np.float64)
		inv_K = np.linalg.inv(self.K).astype(np.float64)
		RT_full = np.concatenate([self.RT,np.array([[0,0,0,1]])], axis=0).astype(np.float64)
		inv_RT_full = np.linalg.inv(RT_full)
		cam_center_world = inv_RT_full[:3,3].reshape(3)
		print('rendering...')
		for x in range(self.width):
		# for x in range(1143,1144):
		# for x in range(600,1000):
			if x % 50 == 0:
				print('Progress: %f%% (%d/%d)' % (float(100*x)/self.width,x,self.width))
			for y in range(self.height):
			# for y in range(220,750):
			# for y in range(527,533):
				pt2d = np.array([x,y], dtype=np.float64)
				pt2d = cv2.undistortPoints(pt2d.reshape(1,1,2), self.K, self.dist, None, self.K)
				pt2d = pt2d.reshape(2)
				x_corrected = pt2d[0]
				y_corrected = pt2d[1]
				trace_loc_cam = np.matmul(inv_K, np.array([x_corrected,y_corrected,1]))
				trace_loc_world = np.matmul(inv_RT_full, np.array(trace_loc_cam.tolist()+[1]))
				trace_loc_world = trace_loc_world[:3]
				pt_start = cam_center_world
				unit_vec = trace_loc_world-cam_center_world
				unit_vec = unit_vec / np.linalg.norm(unit_vec)
				color = scene.primary_ray_tracing(pt_start, unit_vec)
				img_float[y,x] = color
				# print('image x, y', x, y)
				# print('corrected x, y', x_corrected, y_corrected)
				# print('pt_start', pt_start)
				# print('unit_vec', unit_vec)
				# print('color', color)
				# exit()
		return img_float
	def __str__(self):
		ret = ''
		ret += 'K\n'
		ret += str(self.K) + '\n'
		ret += 'RT\n'
		ret += str(self.RT) + '\n'
		ret += 'dist\n'
		ret += str(self.dist) + '\n'
		return ret

def main():
	cam_model_npz = 'camParam-1612675835485.npz'
	data = np.load(cam_model_npz)
	distL = data['distL']
	distR = data['distR']
	L_K = data['intrinsicL']
	R_K = data['intrinsicR']
	L_RT = data['L_RT']
	# L_RT = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,-325]], dtype=np.float64)
	R_RT = data['R_RT']
	Lcamsn = data['Lcamsn']
	Rcamsn = data['Rcamsn']
	width = 2048
	height = 2048
	# rd = MyRenderer(L_K, L_RT, distL, width, height)
	rd = MyRenderer(R_K, R_RT, distR, width, height)
	sc = MyScene()

	# obj = MyObject()
	# obj.set_plane(0.,0.,1.,0)
	# obj.set_phong_material(ks=0.0, kd=0.1, ka=0.05, alpha=1)
	# sc.add_object(obj)

	# obj = MyObject()
	# obj.set_triangle(0,0,0,0,5,0,5,0,0)
	# obj.set_phong_material(ks=0.2, kd=0.2, ka=0.2, alpha=40)
	# sc.add_object(obj)
	# obj = MyObject()
	# obj.set_triangle(5,5,0,0,5,0,5,0,0)
	# obj.set_phong_material(ks=10, kd=10, ka=10, alpha=40)
	# sc.add_object(obj)


	# cube
	# obj = MyObject()
	# obj.set_triangle(0,0,0,0,5,0,5,0,0)
	# obj.set_phong_material(ks=0.2, kd=0.2, ka=0.2, alpha=40)
	# sc.add_object(obj)
	# obj = MyObject()
	# obj.set_triangle(5,5,0,0,5,0,5,0,0)
	# obj.set_phong_material(ks=0.2, kd=0.2, ka=0.2, alpha=40)
	# sc.add_object(obj)

	# obj = MyObject()
	# obj.set_triangle(0,0,5,0,5,5,5,0,5)
	# obj.set_phong_material(ks=0.2, kd=0.2, ka=0.2, alpha=40)
	# sc.add_object(obj)
	# obj = MyObject()
	# obj.set_triangle(5,5,5,0,5,5,5,0,5)
	# obj.set_phong_material(ks=0.2, kd=0.2, ka=0.2, alpha=40)
	# sc.add_object(obj)

	# obj = MyObject()
	# obj.set_triangle(0,0,0,0,5,0,0,0,5)
	# obj.set_phong_material(ks=0.2, kd=0.2, ka=0.2, alpha=40)
	# sc.add_object(obj)
	# obj = MyObject()
	# obj.set_triangle(0,5,5,0,5,0,0,0,5)
	# obj.set_phong_material(ks=0.2, kd=0.2, ka=0.2, alpha=40)
	# sc.add_object(obj)

	# obj = MyObject()
	# obj.set_triangle(5,0,0,5,5,0,5,0,5)
	# obj.set_phong_material(ks=0.2, kd=0.2, ka=0.2, alpha=40)
	# sc.add_object(obj)
	# obj = MyObject()
	# obj.set_triangle(5,5,5,5,5,0,5,0,5)
	# obj.set_phong_material(ks=0, kd=0.2, ka=0.2, alpha=40)
	# sc.add_object(obj)
	# obj = MyObject()
	# obj.set_triangle(0,0,0,0,0,5,5,0,0)
	# obj.set_phong_material(ks=0, kd=0.2, ka=0.2, alpha=40)
	# sc.add_object(obj)
	# obj = MyObject()
	# obj.set_triangle(5,0,5,0,0,5,5,0,0)
	# obj.set_phong_material(ks=0, kd=0.2, ka=0.2, alpha=40)
	# sc.add_object(obj)
	# obj = MyObject()
	# obj.set_triangle(0,5,0,0,5,5,5,5,0)
	# obj.set_phong_material(ks=0, kd=0.2, ka=0.2, alpha=40)
	# sc.add_object(obj)
	# obj = MyObject()
	# obj.set_triangle(5,5,5,0,5,5,5,5,0)
	# obj.set_phong_material(ks=0, kd=0.2, ka=0.2, alpha=40)
	# sc.add_object(obj)


	# obj = MyObject()
	# obj.set_sphere(9.,7.,-1.,5.)
	# obj.set_phong_material(ks=100., kd=0.2, ka=0.2, alpha=40)
	# sc.add_object(obj)

	obj = MyObject()
	obj.set_sphere(8.,8.,0.,5.)
	obj.set_phong_material(ks=0.5, kd=0.2, ka=0.2, alpha=40)
	sc.add_object(obj)

	obj = MyObject()
	obj.set_ambient_light(0.5)
	sc.add_object(obj)

	obj = MyObject()
	obj.set_point_light(30.,-30.,-10.,1.,1.)
	sc.add_object(obj)

	# ring light
	# obj = MyObject()
	# obj.set_point_light(30.,-30.,-10.,1.,1.)
	# sc.add_object(obj)
	# obj = MyObject()
	# obj.set_point_light(0.,-30.,-10.,1.,1.)
	# sc.add_object(obj)
	# obj = MyObject()
	# obj.set_point_light(-30.,-30.,-10.,1.,1.)
	# sc.add_object(obj)
	# obj = MyObject()
	# obj.set_point_light(-30.,0.,-10.,1.,1.)
	# sc.add_object(obj)
	# obj = MyObject()
	# obj.set_point_light(-30.,30.,-10.,1.,1.)
	# sc.add_object(obj)
	# obj = MyObject()
	# obj.set_point_light(0.,30.,-10.,1.,1.)
	# sc.add_object(obj)
	# obj = MyObject()
	# obj.set_point_light(30.,30.,-10.,1.,1.)
	# sc.add_object(obj)
	# obj = MyObject()
	# obj.set_point_light(30.,0.,-10.,1.,1.)
	# sc.add_object(obj)

	# # ring light v2
	# ring_light_z = -10
	# ring_light_max_x = 30
	# ring_light_min_x = -30
	# ring_light_max_y = 30
	# ring_light_min_y = -30
	# light_step = 5
	# xs = [x for x in range(ring_light_min_x, ring_light_max_x + light_step, light_step)]
	# ys = [y for y in range(ring_light_min_y + light_step, ring_light_max_y, light_step)]
	# # print(xs)
	# # print(ys)
	# ring_light_num_lights = 2*(len(xs) + len(ys))
	# light_I = 2.0 / float(ring_light_num_lights)
	# for x in xs:
	# 	obj = MyObject()
	# 	obj.set_point_light(x, ring_light_min_y, ring_light_z, light_I, light_I)
	# 	sc.add_object(obj)
	# 	obj = MyObject()
	# 	obj.set_point_light(x, ring_light_max_y, ring_light_z, light_I, light_I)
	# 	sc.add_object(obj)
	# for y in ys:
	# 	obj = MyObject()
	# 	obj.set_point_light(ring_light_min_x, y, ring_light_z, light_I, light_I)
	# 	sc.add_object(obj)
	# 	obj = MyObject()
	# 	obj.set_point_light(ring_light_max_x, y, ring_light_z, light_I, light_I)
	# 	sc.add_object(obj)
	# print(sc)
	# exit()

	# structured light v1
	# for x in np.linspace(-3,3,13):
	# 	for y in np.linspace(-3,3,13):
	# 		obj = MyObject()
	# 		obj.set_cylinder_light(x, y, -200, 0, 0, 1, 0.1, 0.5, 0.5)
	# 		sc.add_object(obj)
	# print(sc)
	# exit()
	# obj = MyObject()
	# obj.set_cylinder_light(8, 8, -200, 0, 0, 1, 0.1, 0.5, 0.5)
	# sc.add_object(obj)

	obj = MyObject()
	obj.set_cone_light(8, 8, -200, 0, 0, 1, 1, 50, 50)
	sc.add_object(obj)

	timestamp = current_milli_time()
	scene_fn = 'scene-%d.txt' % timestamp
	with open(scene_fn, 'w') as f:
		f.write(str(rd))
		f.write(str(sc))
	print('save scene to', scene_fn)

	img_float = rd.render_primary_ray_tracing(sc)
	img_uint8 = img_float_to_uint8(255.0*img_float)
	img_fn = 'tmp-%d.png' % timestamp
	cv2.imwrite(img_fn, img_uint8)
	print('saving image to', img_fn)
	cv2.imshow('img', cv2.resize(img_uint8,(800,800)))
	cv2.waitKey(0)

if __name__ == "__main__":
	main()