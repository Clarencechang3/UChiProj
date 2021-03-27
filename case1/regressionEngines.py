import numpy as np
import matplotlib.pyplot as plt
 

#Writing good code here
class reg_module():
    def __init__(self, revenue_list):
        self.type = "None"
        self.coefs = np.array([[1], [1]])
    def evaluate(self, x):
        phi = np.array([[i, 1] for i in x])
        pred = np.dot(phi, self.coefs)
        return pred
    def qr_reg (self, phi, y):
        Q, R = np.linalg.qr(phi)

        b_bar = np.dot(Q.transpose(),y)

        backsub = self.backwardsub(R, b_bar)
        
        self.coefs = backsub
    def backwardsub(self, U, b):
        
        n = U.shape[1]
        star = list()
        star.append(b[-1]/U[-1, -1])
        for i in range(n-2, -1, -1):
        
            diag_val = U[i, i]
            y = b[i]
            pre_computed = np.array(star[::-1])
            U_greater_than = U[i, i:-1]
            dp = np.dot(U_greater_than, pre_computed)
            numerator = y - dp
            ans = numerator/diag_val
            star.append(ans)
        star = np.array(star[::-1])
        return star   
##############################################################################
class exp_reg(reg_module):
    def __init__(self, revenue_list):
        reg_module.__init__(self, revenue_list)
        self.type = "Expotential"
        log_revenue = np.array([[np.log(i)] for i in revenue_list])
        phi = np.array([[i, 1] for i in range(1, len(log_revenue)+1)])
        self.qr_reg(phi, log_revenue)
   
    def backwardsub(self, U, b):
        #I have no idea why this works
        n = U.shape[0]-1
        star = list()
        star.append(b[n]/U[n, n])
        star.append((b[n-1] - np.dot(U[n-1, n], star[0]))/U[n-1, n-1])
        star = np.array(star[::-1])
        return star    
    def evaluate(self, x):
        phi = np.array([[i, 1] for i in x])
        pred = np.dot(phi, self.coefs)
        plt.figure()
        plt.plot(pred)
        pred = np.exp([x for x in pred])
        return pred

    
class lin_reg(reg_module):
    def __init__(self, revenue_list):
        reg_module.__init__(self, revenue_list)
        self.type = "Linear"
        revenue = np.array([[i] for i in revenue_list])
        phi = np.array([[i, 1] for i in range(1, len(revenue_list)+1)])
        self.qr_reg(phi, revenue)

class quad_reg(reg_module):
    def __init__(self, revenue_list):
        reg_module.__init__(self, revenue_list)
        self.type = "Quadratic"
        revenue = np.array([[i] for i in revenue_list])
        phi = np.array([[i**2, i, 1] for i in range(1, len(revenue_list)+1)])
        self.qr_reg(phi, revenue)
    def evaluate(self, x):
        phi = np.array([[i**2, i, 1] for i in x])
        pred = np.dot(phi, self.coefs)
        return pred
