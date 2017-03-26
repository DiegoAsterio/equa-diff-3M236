import numpy as np
import matplotlib.colors as colors
import matplotlib.cm as cmx
import math

def get_cmap(N):
    '''Returns a function that maps each index in 0, 1, ... N-1 to a distinct 
    RGB color.'''
    color_norm  = colors.Normalize(vmin=0, vmax=N)
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='hsv')
    def map_index_to_rgb_color(index):
        return scalar_map.to_rgba(index)
    return map_index_to_rgb_color

def stormer_verlet(t0, x0, dt, T, gamma):
    t_ = np.arange(t0, T + dt, dt)

    f= gamma[0]
    g= gamma[1]
    
    un = x0[0]
    vn = x0[1]
    x_ = [np.array(x0)]

    for t in t_[:-1]:
        aux = vn + 0.5*dt*g(un,t)
        un = un + dt*f(aux,t)
        vn = aux + 0.5*dt*g(un,t)
        val = np.array([un, vn])
        x_ = np.concatenate((x_, [val]))

    return t_, x_

def euler_explicite(t0, x0, dt, T, f):
    t_ = np.arange(t0, T + dt, dt)
    x_ =[np.array(x0)]
    
    for t in t_[:-1]:
        val = x_[-1] + dt*f(x_[-1],t)
        val = np.array(val)
        x_ = np.concatenate((x_, [val]))
    
    return t_, x_

def dist_euclidienne(x, y):
    v = [(i-j)*(i-j) for i,j in np.column_stack([x,y])]
    return np.sqrt(np.sum(v))
    

def RK23(t0, x0, dt_init, dt_min, T, eps, f):
    n_fail = 0
    tn = t0

    dt = dt_init
    x_ = [np.array(x0)]
    t_ = [t0]
    
    while tn < T and n_fail < 1000:
        k1 = f(x_[-1],tn)
        k2 = f(x_[-1]+dt*k1, tn + dt)
        k3 = f(x_[-1]+0.25*dt*(k1 + k2) , tn+ 0.5*dt)

        tau = dt*dist_euclidienne(2*k3, k1+k2)/3

        if tau < eps:
            val = x_[-1] + dt*(k1 + k2 + 4*k3)/6
            val = np.array(val)
            x_ = np.concatenate((x_,[val]))
            tn = tn + dt
            t_.append(tn)
            if tau < 0.1*eps:
                dt = 2*dt
        else:
            dt = max(0.5*dt, dt_min)
            n_fail += 1

    return t_ , x_

def separate_solutions(t_,u_,v_,tini,tfin): #Pour separer les solutions pour un intervalle de temps
    tret = []
    uret = []
    vret = []
    for i in range(len(t_)):
        if t_[i] >= tini and t_[i]<=tfin:
            tret.append(t_[i])
            uret.append(u_[i])
            vret.append(v_[i])
    return tret, uret, vret


