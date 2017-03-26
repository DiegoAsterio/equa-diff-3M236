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

def EulerSymplectique(t0, x0, T, dt, f):
    # x0 et f(x0, t0) doivent etre des vecteurs lignes
    # renvoie TPS et X
    # TPS est un vecteur ligne contenant les (N+1) temps intermediaires (t0, ..., tN)
    # X a autant de colonnes que X0 et N+1 lignes; le résultat au temps ti est sur la ligne (i+1)
    if np.size(x0)==1: # on regarde si x0 est un reel
        # dans ce cas on le transforme en vecteur ligne de taille 1
        x0=np.array([float(x0)])

    # initialisation
    TPS=[t0] # TPS est un vecteur ligne
    X=[x0] #TPS est un tableau avec le même nombre de colonne que x0, et une ligne pour chaque t_i

    while t0 < T:
        # calcul de ma solution au temps suivant
        t1=t0+dt
        aux1=x0+dt*f(x0,t0)
        aux2=x0+dt*f([aux1[0],x0[1]],t1)
        x1 = np.asarray([aux1[0],aux2[1]])
        
        # stockage des resultats
        TPS=np.append(TPS,[t1])
        X=np.concatenate((X,[x1]))
        
        # actualisation des valeurs
        t0=t1
        x0=x1
    return TPS, X



