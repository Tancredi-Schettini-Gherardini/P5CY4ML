# -*- coding: utf-8 -*-


# Import libraries

import numpy as np
from itertools import product

#%% --------------------------------------------------------------------------------------
#   We define the functions needed for the implementation of the Hodge numbers
#   calculation based on the Landau-Ginzburg model.


#  We start by defining the basic quantities and operations involved in the computation.

def theta(q,l): 
    # Finds \theta(q), defined in arXiv:2006.15825. q should be a vector and l should be an integer.
    
    theta = q*l 
    return theta
    
def theta_bar(q, l): 
    # Finds \bar{theta}(q), defined in arXiv:2006.15825. q should be a vector and l should be an integer
    
    the = theta(q,l)
    theta_bar = np.zeros(len(the))
    
    for i in range(0,len(the)):
        if abs(the[i]-round(the[i]))<0.000001:
            theta_bar[i] = 0
        else:
            theta_bar[i] = the[i] - np.floor(the[i])
        
    return theta_bar

def age(q,l):
    # Finds age(q), defined in arXiv:2006.15825. q should be a vector and l should be an integer
    
    theta_b = theta_bar(q,l)
    age=0
    for p in range(0,len(theta_b)):
        age = age + theta_b[p]
    
    return age

def size(q,l,tot_w):
    # Finds size(q), defined in arXiv:2006.15825. q should be a vector and l should be an integer
    size = age(q, l) + age(q, tot_w - l)
    
    return size


def Product(a,b): 
    # a,b must be a list whose entries are of the form (coefficient, exponent). 
    # This function returns the product of the two polynomials, in the same format.
    
    if len(b)==0:
        
        return a
    
    else:
        
        c=list(product(a,b))
        c=np.array(c)
        raw_product = np.zeros((len(c),2))
        for i in range(len(c)):
            raw_product[i,0] = c[i,0,0] * c[i,1,0]
            raw_product[i,1] = c[i,0,1] + c[i,1,1]
            
        for i in range(len(c)):
            
            for j in range (i+1,len(raw_product)):

                if j<len(raw_product):
                    if abs(raw_product[i,1] - raw_product[j,1])<0.0000000001:
                        raw_product[i,0] = raw_product[i,0] + raw_product[j,0]
                        raw_product = np.delete(raw_product, j, 0)
                
                
        return raw_product


def Find_fractions(q, theta, d, l): 
    #Given q, \theta(q), d = len(q) and some integer l, it finds the lth contribution
    # to the Hodge numbers' formula in terms of the polynomial
    # fractions involved (corollary 4.4 in arXiv:2006.15825).
    first_factor = 0
    fractions = []
    ct = 0
    for j in range(0,d):
        
        if theta_bar(q,l)[j] == 0:
            
            ct += 1
            first_factor = first_factor + q[j]
            fractions.append([[1,0],[-1,1-q[j]]])
            fractions.append([[1,0],[-1,q[j]]])
    
    if ct==0:
        fractions.append([[1,0],[1,1]])
        fractions.append([[1,0],[1,1]])
    
    return np.array(fractions), first_factor

def Total_quotient_clean(fractions,w_tot):
    # Given a number of fractions, it multiplies them together.
    # It first calculates the product of all the numerators, then the product of
    # all the denominators, and then takes the quotient.
    c_int_w = 0
    count_w = []
    fr_index = []

    for i in range(round(len(fractions)/2)):
        
        if abs(fractions[2*i,1,1] - fractions[(2*i+1),1,1]) < 0.0000001:
            c_int_w += 1
        
        else:
            count_w.append(c_int_w)
            c_int_w=1
            fr_index.append(i-1)   
            
    n_diff_frac = len(count_w)
    
    numers = []
    denoms = []
    integer_exp_num = []
    integer_exp_den = []
    px = []
    gx = []
    px_multiple = []
    px_multiple = []
    
    n_diff_frac = round(len(fractions)/2)

    for i in range(round(len(fractions)/2)):
        
        numers.append(fractions[2*i])
        denoms.append(fractions[2*i + 1])
        integer_exp_num.append(round(w_tot*fractions[2*i,1,1]))
        integer_exp_den.append(round(w_tot*fractions[2*i + 1,1,1]))
        
        px.append(-1)
        for j in range(integer_exp_num[i]-1):
            px.append(0)
        px.append(1)
        
        gx.append(-1)
        for j in range(integer_exp_den[i] - 1):
            gx.append(0)
        gx.append(1)

    
    px_tot = px[0:integer_exp_num[0]+1]
    gx_tot = gx[0:integer_exp_den[0]+1]
    
    start_n = integer_exp_num[0]+1
    start_d = integer_exp_den[0]+1
    
    for i in range(1,round(len(fractions)/2)):
        px_tot = np.polymul(px_tot, px[start_n: start_n + integer_exp_num[i] + 1])
        gx_tot = np.polymul(gx_tot, gx[start_d: start_d + integer_exp_den[i] + 1])
        start_n = start_n + integer_exp_num[i] + 1
        start_d = start_d + integer_exp_den[i] + 1
    
    qx_tot, rx_tot = np.polydiv(px_tot,gx_tot)
    
    if any(rx_tot):
        # There should be no reminder in the polynomial division.
        print("Warning")
    
    return qx_tot 


def Hodge_reader_3folds(poly):
    
    # It just reads the Hodge numbers off from the polynomial, for C-Y three-folds.
    
    h11 = 0
    h12 = 0
    
    for J in range(len(poly)):
        
        if J<len(poly) and abs(poly[J,1]-1)<0.000001 and abs(poly[J,2]-1)<0.00001:
            h11=(poly[J,0])
            
        if J<len(poly) and abs(poly[J,1]-1)<0.000001 and abs(poly[J,2]-2)<0.00001:
            h12=-(poly[J,0])
            
    return [h11, h12]
        


def Hodge_reader_4folds(poly):
    
    # It just reads the Hodge numbers off from the polynomial, for C-Y four-folds.
    
    h11 = 0
    h12 = 0
    h13 = 0
    h22 = 0
    
    for J in range(len(poly)):
        
        if J<len(poly) and abs(poly[J,1]-1)<0.000001 and abs(poly[J,2]-1)<0.00001:
            h11=(poly[J,0])
            
        if J<len(poly) and abs(poly[J,1]-1)<0.000001 and abs(poly[J,2]-2)<0.00001:
            h12=-(poly[J,0])
    
        if J<len(poly) and abs(poly[J,1]-1)<0.000001 and abs(poly[J,2]-3)<0.00001:
            h13=(poly[J,0])
            
        if J<len(poly) and abs(poly[J,1]-2)<0.000001 and abs(poly[J,2]-2)<0.00001:
            h22=(poly[J,0])
    
    return [h11, h12, h13, h22]


#%% Now we define the main functions: one for the calculation of the exact
#   Hodge numbers, and the other one for the calculation of the approximated ones.

def Poincare_clean(w): 
    """

    Parameters
    ----------
    w : 1-dimensional np.array of integers
        The input is a (Calabi-Yau) weight-system of any dimension.

    Returns
    -------
    summand : np.array with 3 columns.
    
        It is the Poincare-like polynomial which encodes the Hodge 
        numbers. The first column contains the coefficients, while the other two
        columns contain the powers of the variables u and v (see corollary 4.4 
        in arXiv:2006.15825). To read off the Hodge numbers from this array, one
        can use the "Hodge_reader" functions above.
        
    approximation : np.array with 3 columns.
    
        It is the polynomial which encodes the approximated Hodge numbers. 
        The first column contains the coefficients, while the other two
        columns contain the powers of the variables u and v. 
        To read off the approximated Hodge numbers from this array, one
        can use the "Hodge_reader" functions above.

    """
    
    d = len(w)
    tot_w = 0
    
    for i in range(0,d):
        tot_w = tot_w + w[i]
        
    q = w / tot_w
    
    approximation = []
    summand=[]
    
    for l in range(0,tot_w):
        # We perform the sum in Batyrev formula, term by term.
        simp_fr = []
        quotient = []
        first_f = 0
        
        theta_b = theta_bar(q,l)
         
        fr, first_f = Find_fractions(q, theta(q,l), d, l) 
        # This finds the fractions that we need to multiply for the current term.

        if len(fr)>0:
        
            quotient = Total_quotient_clean(fr, tot_w)

            quoti = []
            
            for i in range(len(quotient)):
                # We append the powers next to the coefficients.
            
                quoti.append([quotient[i], (len(quotient) - i -1)/tot_w + first_f])

            final_prod = quoti
            
        else: 
            final_prod = []
            
        final_prod = np.array(final_prod)
        
        final_prod_int = []
        
        # We now select the integer coefficients. Note that if there are none,
        # the contribution is taken to be 1.
        
        if final_prod.size == 0:
            
            summand.append([(-1)**(round(size(q,l,tot_w))),   age(q,l) - 1, size(q,l,tot_w) - age(q,l) - 1])
        
        elif len(np.array(final_prod).shape)==1:
         
                if abs(final_prod[1] - round(final_prod[1])) < 0.0000001:
                    
                    if final_prod[i,0] > 0:
                    
                        final_prod_int.append([final_prod[0], final_prod[1]])
                        
        elif len(np.array(final_prod).shape)>1:
            
            for i in range(len(final_prod)):
                
                if abs(final_prod[i,1] - round(final_prod[i,1])) < 0.0000001:
                
                    if final_prod[i,0] > 0.0000001:
                
                        final_prod_int.append([final_prod[i,0], final_prod[i,1]])

        final_prod_int = np.array(final_prod_int)

        for i in range(len(final_prod_int)):
            # We multiply the current result by the appropriate powers of u and v,
            # again according to corollary 4.4 in arXiv:2006.15825.

            summand.append([final_prod_int[i,0] * (-1)**(round((size(q,l,tot_w)))), final_prod_int[i,1] + age(q,l) - 1, final_prod_int[i,1] + size(q,l,tot_w) - age(q,l) - 1])
            
            if round(size(q,l,tot_w))==d:
                # This is one of the two terms contributing to our approximation.
                
                approximation.append([final_prod_int[i,0] * (-1)**(round((size(q,l,tot_w)))), final_prod_int[i,1] + age(q,l) - 1, final_prod_int[i,1] + size(q,l,tot_w) - age(q,l) - 1])   
                

        if l==0:
            # This is the other of the two terms contributing to our approximation.

            for i in range(len(summand)):
                approximation.append(summand[i])  
               
        
    summand=np.array(summand)
    for k in range(10):
        # Here we just make sure that all the terms with the same power have been summed up.
        
        for i in range(len(summand)):
            
            for j in range (i+1,len(summand)):

                if j<len(summand):
                    if abs(summand[i,1] - summand[j,1])<0.000001 and abs(summand[i,2] - summand[j,2])<0.000001:
                        summand[i,0] = summand[i,0] + summand[j,0]
                        summand = np.delete(summand, j, 0)
                    
    approximation=np.array(approximation)   

    for k in range(10):
        # Here we just make sure that all the terms with the same power have been summed up.
        
        for i in range(len(approximation)):
            
            for j in range (i+1,len(approximation)):

                if j<len(approximation):
                    if abs(approximation[i,1] - approximation[j,1])<0.000001 and abs(approximation[i,2] - approximation[j,2])<0.000001:
                        approximation[i,0] = approximation[i,0] + approximation[j,0]
                        approximation = np.delete(approximation, j, 0)
    
  
    return summand, approximation



def Poincare_approx_clean(w):

    """
    
    Parameters
    ----------
    w : 1-dimensional np.array of integers
        The input is a (Calabi-Yau) weight-system of any dimension.

    Returns
    -------
        
    approximation : np.array with 3 columns.
    
        It is the polynomial which encodes the approximated Hodge numbers. 
        The first column contains the coefficients, while the other two
        columns contain the powers of the variables u and v. 
        To read off the approximated Hodge numbers from this array, one
        can use the "Hodge_reader" functions above.

    """
    
    d = len(w)
    tot_w = 0
    
    for i in range(0,d):
        tot_w = tot_w + w[i]
    q = w / tot_w
    
    approximation = []
    summand=[]
    counter_size4=0
    counter_size3=0
    
    
    for l in range(0,tot_w):
        
        if round(size(q,l,tot_w))==d:
            # This founds one contribution, corresponding to the maximum size elements,
            # which involves no polynomial division.
            
            approximation.append([(-1)**(round((size(q,l,tot_w)))),  + age(q,l) - 1,  + size(q,l,tot_w) - age(q,l) - 1])
        
        if l==0:
            # The other contribution comes from the l=0 term in the sum.
            
            simp_fr = []
            extra_t = []
            quotient = []
            first_f = 0
            
            theta_b = theta_bar(q,l)
             
    
            fr, first_f = Find_fractions(q, theta(q,l), d, l)
            # This finds the fractions that we need to multiply for the current term.
    
            if len(fr)>0:
            
                quotient = Total_quotient_clean(fr, tot_w)
    
                quoti = []
                
                for i in range(len(quotient)):
                    # We append the powers next to the coefficients.
                
                    quoti.append([quotient[i], (len(quotient) - i -1)/tot_w + first_f])
    
                final_prod = Product(quoti, extra_t)
                
            
            else: 
                final_prod = []
                
            final_prod = np.array(final_prod)
            
            final_prod_int = []
            
            # We now select the integer coefficients. Note that if there are none,
            # the contribution is taken to be 1.
            if final_prod.size == 0:
                
                summand.append([(-1)**(round(size(q,l,tot_w))),   age(q,l) - 1, size(q,l,tot_w) - age(q,l) - 1])
            
            elif len(np.array(final_prod).shape)==1:
             
                    if abs(final_prod[1] - round(final_prod[1])) < 0.0000001:
                        
                        if final_prod[i,0] > 0:
                        
                            final_prod_int.append([final_prod[0], final_prod[1]])
                            
            elif len(np.array(final_prod).shape)>1:
                
                for i in range(len(final_prod)):
                    
                    if abs(final_prod[i,1] - round(final_prod[i,1])) < 0.0000001:
                    
                        if final_prod[i,0] > 0.0000001:
                    
                            final_prod_int.append([final_prod[i,0], final_prod[i,1]])

            final_prod_int = np.array(final_prod_int)

            for i in range(len(final_prod_int)):
                # We multiply the current result by the appropriate powers of u and v,
                # again according to corollary 4.4 in arXiv:2006.15825.
                
                summand.append([final_prod_int[i,0] * (-1)**(round((size(q,l,tot_w)))), final_prod_int[i,1] + age(q,l) - 1, final_prod_int[i,1] + size(q,l,tot_w) - age(q,l) - 1])
                
                #######if round(size(q,l,tot_w))==d:
                    
                    ###approximation.append([final_prod_int[i,0] * (-1)**(round((size(q,l,tot_w)))), final_prod_int[i,1] + age(q,l) - 1, final_prod_int[i,1] + size(q,l,tot_w) - age(q,l) - 1])   
        
            for i in range(len(summand)):
                approximation.append(summand[i])  
               
    approximation=np.array(approximation)   

    for k in range(10):
        # Here we just make sure that all the terms with the same power have been summed up.
        
        for i in range(len(approximation)):
            
            for j in range (i+1,len(approximation)):

                if j<len(approximation):
                    if abs(approximation[i,1] - approximation[j,1])<0.000001 and abs(approximation[i,2] - approximation[j,2])<0.000001:
                        approximation[i,0] = approximation[i,0] + approximation[j,0]
                        approximation = np.delete(approximation, j, 0)
    
  
    return approximation