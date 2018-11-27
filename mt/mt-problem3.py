
# coding: utf-8

# In[1]:


import numpy as np


# In[4]:


def nonempty_count(M):
    count = 0
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if M[i, j] == -1:
                count += 1
    return M.size - count


def init_u_v(M, d):
    count = 0
    sum = 0
    
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            
            if not M[i, j] == -1:
                sum += M[i,j]
                count += 1
         
    avg = np.sqrt((sum / count) / d)

    m, n = M.shape
    
    u = np.ones([m, d]) * avg
    v = np.ones([d, n]) * avg
    return u, v
    


# In[5]:


def RMSE(T, M):
    
#     T = np.dot(u, v)
    
    error = 0
    for j in range(M.shape[1]):
        
        row_error = 0
        for i in range(M.shape[0]):
            if M[i][j] == -1:
                continue
            row_error += (M[i, j] - T[i, j])**2
            
        error += row_error
        
    return np.sqrt(error / nonempty_count(M)) 


# In[6]:


# u, v = init_u_v(M, 3)
# # print(u, v)
# T = np.dot(u, v)
# RMSE(T, M)


# In[7]:


u = np.ones([5, 2])
v = np.ones([2, 5])


# In[18]:


def solve_U_at(r, s, u, v, M):
    denom = 0 
    numer = 0
    
    for j in range(v.shape[1]):
        
        if M[r, j] == -1:
            continue

        if v[s, j] != -1:
            denom += v[s, j]**2
    
    
    for j in range(v.shape[1]):
        if M[r, j] == -1:
            continue
        
        sum1 = 0
        for k in range(u.shape[1]):
            if k == s:
                continue
            sum1 += u[r, k] * v[k, j]
    
        numer += v[s, j] * (M[r, j] - sum1)
        
    
    return numer/denom
    
    


# In[19]:


def solve_V_at(r, s, u, v, M):
    numer = 0
    denom = 0
    
        
    
    for i in range(u.shape[0]):
        if M[i, s] == -1:
            continue            
        if u[i,r] != -1:
            denom += u[i, r]**2
            
    
    for i in range(u.shape[0]):
        if M[i, s] == -1:
            continue            
        
        sum1 = 0
        for k in range(v.shape[0]):
            if k == r:
                continue
            sum1 += u[i, k] * v[k, s]
        
        numer += u[i, r] * (M[i, s] - sum1)
        
    return numer/denom
    


# In[20]:



def solve_U(u, v, M):
    for r in range(u.shape[0]):
        for s in range(u.shape[1]):
            u[r, s] = solve_U_at(r, s, u, v, M)
            
    return u

def solve_V(u, v, M):
    for r in range(v.shape[0]):
        for s in range(v.shape[1]):
            v[r, s] = solve_V_at(r, s, u, v, M)
            
    return v



# In[21]:


# Book examples 

# u = np.ones([5, 2])
# v = np.ones([2, 5])

# M = np.array([
#     [5, 2, 4, 4, 3],
#     [3, 1, 2, 4, 1], 
#     [2, -1, 3, 1, 4], 
#     [2, 5, 4, 3, 5], 
#     [4, 4, 5, 4, -1], 
# ])
# 


# In[22]:


M = np.array([
    [2, 1, -1, 1, -1, 5],
    [4, -1, 2, -1, -1, -1], 
    [3, 3, -1, 5, 1, -1], 
    [-1, -1, 5, -1, 1, 2], 
])

u, v = init_u_v(M, 3)
print(RMSE(np.dot(u, v), M))

for i in range(10):
    u = solve_U(u, v, M)
    v = solve_V(u, v, M)
    
M_hat = np.dot(u, v)
print(RMSE(M_hat, M))


# In[23]:


u


# In[24]:


v


# In[25]:


M_hat

