from typing import List
import math

class Solution:
    def infinite_rotation(self, points, t):
        res = []
        for p in points:
            res.append(self.convergence(p, t))

        return res
    
    def convergence(self, points, t):
        n = len(points)
        pts = [p[:] for p in points]
        alpha = 0.3
        eps = 1e-6

        for _ in range(t):
            new_pts = [[0,0] for _ in range(n)]

            for i in range(n):
                j = (i+1) % n
                xi, yi = pts[i]
                xj, yj = pts[j]

                new_pts[i][0] = (1-alpha)*xi + alpha*xj
                new_pts[i][1] = (1-alpha)*yi + alpha*yj

            pts = new_pts

            if self.max_dis(pts) <= eps:
                return True
            
        return False
            
    
    def max_dis(self, points):
        n = len(points)
        max_dis = 0
        for i in range(n):
            x1, y1 = points[i]
            for j in range(i+1, n):
                x2, y2 = points[j]

                dx = x1 - x2
                dy = y1 - y2

                d = math.hypot(dx, dy)
                if d > max_dis:
                    max_dis = d

        return max_dis


# sol = Solution()

# systems = [[[1,0], [0,1], [-1,0], [0,-1]], [[0,0], [10,0]]]

# result = sol.infinite_rotation(systems, 50)
# print(result)

class Solution2:
    def binary_func(self, s, base):
        b_map = {"bin" : 2, "dec" : 10, "oct": 8, "hex" : 16}

        val = ''.join(str(d) for d in s)

        dec_val = int(val, b_map[base])

        return [bin(dec_val)[2:], oct(dec_val)[2:], dec_val, hex(dec_val)[2:]]
    
sol = Solution2()
s = [1, 0, 0, 1, 1]
base = "hex" 

sol = print(sol.binary_func(s, base))
print(sol)