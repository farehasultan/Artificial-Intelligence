'''
Fareha Sultan
100968491

This program recognizes letters A , B, C, D, E from black and white
images using two different classifiers: a Naive Bayes Classifier and 
a Fuzzy Classifier.

We will extract the following features from the images:
1. PropBlack : The proportion of pixels in the image that are black.
2. TopProp: The proportion of the black pixels that are in the top half of the image.
3. LeftProp: The proportion of the black pixels that are in the left half of the image.

The image will be in a matrix representation where "1" is black and "0" is white.

A , B, C, D , E are classes.

The general formula for the probability density function of the normal distribution is
f(x)=e−(x−μ)2/(2σ2)σ2π√ where μ is the location parameter and σ is the scale parameter.
https://www.itl.nist.gov/div898/handbook/eda/section3/eda3661.htm
μ = mean, standard deviation =σ
'''

import math

#-------------------------Naive Bayes Classifier--------------------------------------------------
classes=[
{ "class": "A",
  "priorProb": 0.28,
  "propBlack_mean": 0.38,
  "propBlack_sd": 0.06,
  "topProp_mean": 0.46,
  "topProp_sd": 0.12,
  "leftProp_mean":0.50,
  "leftProp_sd":0.09
},
{ "class":"B",
  "priorProb": 0.05,
  "propBlack_mean": 0.51,
  "propBlack_sd": 0.06,
  "topProp_mean": 0.49,
  "topProp_sd": 0.12,
  "leftProp_mean":0.57,
  "leftProp_sd":0.09
},
{ "class":"C",
  "priorProb": 0.10,
  "propBlack_mean": 0.31,
  "propBlack_sd": 0.06,
  "topProp_mean": 0.37,
  "topProp_sd": 0.09,
  "leftProp_mean":0.64,
  "leftProp_sd":0.06
},
{ "class":"D",
  "priorProb": 0.15,
  "propBlack_mean": 0.39,
  "propBlack_sd": 0.06,
  "topProp_mean": 0.47,
  "topProp_sd": 0.09,
  "leftProp_mean":0.57,
  "leftProp_sd":0.03
},
{ "class":"E",
  "priorProb": 0.42,
  "propBlack_mean": 0.43,
  "propBlack_sd": 0.12,
  "topProp_mean": 0.45,
  "topProp_sd": 0.15,
  "leftProp_mean":0.65,
  "leftProp_sd":0.09
}]


''' This function is equal to P(feature|class) '''
def normal_probability_density(x,μ,σ):
  p = (1/math.sqrt(2*math.pi*(σ**2)))*(math.e**((-1/2)*(((x-μ)/σ)**2)))
  return p

def blackPixels(image,x,y): #total number of black pixels
  count=0
  for row in range(y):
    for col in range(x):
      if(image[row][col]=='1'):
        count+=1
  return count

def propBlack(x,y,pix):
  totalPixels = x*y
  return pix/totalPixels

def topProp(image,x,y,pix):
  count=0
  topHalf = int(y/2)
  for row in range(topHalf):
    for col in range(x):
      if(image[row][col]=='1'):
        count+=1
  topProp = count/pix
  return topProp

def leftProp(image,x,y,pix):
  count=0
  leftHalf = int(x/2)
  for row in range(y):
    for col in range(leftHalf):
      if(image[row][col]=='1'):
        count+=1
  leftProp = count/pix
  return leftProp

def calcProb(classs,features):
  #Calculating P(feature|class)
  #3 Features: propBlack, topProp, leftProp
  propBlack = normal_probability_density(features["propBlackValue"],classs["propBlack_mean"],classs["propBlack_sd"])
  topProp = normal_probability_density(features["topPropValue"],classs["topProp_mean"],classs["topProp_sd"])
  leftProp = normal_probability_density(features["leftPropValue"],classs["leftProp_mean"],classs["leftProp_sd"])
  priorProb = classs["priorProb"]
  return propBlack * topProp * leftProp * priorProb

def featuresVector(image):
  features = {}
  Y = (len(image)) #The Y value
  X = (len(image[0])) #The X value
  numBlackPixel = blackPixels(image, X, Y)
  features["propBlackValue"] = propBlack(X, Y, numBlackPixel)
  features["topPropValue"] = topProp(image, X, Y, numBlackPixel)
  features["leftPropValue"] = leftProp(image, X, Y, numBlackPixel)
  return features
   
def naive_bayes_classifier(input_filepath):
  # input is the full file path to a CSV file containing a matrix representation of a black-and-white image
  image = [] #Saving the graph from csv file 
  with open(input_filepath) as f: 
      for line in f.readlines():
        image.append(line[:-1].split(',')) 
  f.close()
 
  features = featuresVector(image)

  class_probabilities = {}
  for i in range(len(classes)):
    class_probabilities[classes[i]["class"]] = calcProb(classes[i],features)
  
  addProbs=sum(class_probabilities.values())

  for key, value in class_probabilities.items():
    class_probabilities[key]=(value/addProbs)

  # most_likely_class is a string indicating the most likely class, either "A", "B", "C", "D", or "E"
  most_likely_class=""
  for key, value in class_probabilities.items():
    if (value == max(class_probabilities.values())):
      most_likely_class=key
  
  # class_probabilities is a five element list indicating the probability of each class in the order [A probability, B probability, C probability, D probability, E probability]
  return most_likely_class, list(class_probabilities.values())




#--------------------------Fuzzy Classifier------------------------------------------------------

#For the fuzzy rules we will the Godel t-norm and the Godel s-norm.
def t_norm(x,y):
  return min(x,y)

def s_norm(x,y):
  return max(x,y)

def trapezoidal_function(x,arr):
  a = arr[0]
  b = arr[1]
  c = arr[2]
  d = arr[3]
  if (x<=a):
    return 0
  if(a<x and x < b):
    return (x-a)/(b-a)
  if(b<= x and x <= c):
    return 1
  if(c<x and x<d):
    return (d-x)/(d-c)
  if(d <= x):
    return 0

fuzzySets=[ 
  {"feature":"PropBlack",
  "Low": [0, 0, 0.3, 0.4], 
  "Med": [0.3,0.4,0.4,0.5],
  "High":[0.4, 0.5,1, 1]
  },
  {"feature": "TopProp",
  "Low": [0, 0, 0.3, 0.4],
  "Med": [0.3, 0.4, 0.5, 0.6],
  "High": [0.5, 0.6, 1, 1]
  },
  {"feature": "LeftProp",
  "Low": [0, 0, 0.3, 0.4],
  "Med":[0.3, 0.4, 0.6,0.7],
  "High" : [0.6, 0.7, 1, 1]
  }
]

def fuzzy_classifier(input_filepath):
  # input is the full file path to a CSV file containing a matrix representation of a black-and-white image
  image = [] #Saving the graph from csv file 
  with open(input_filepath) as f: 
      for line in f.readlines():
        image.append(line[:-1].split(',')) 
  f.close()
 
  features = featuresVector(image)

  #Rule 1: IF PropBlack is Medium AND (TopProp is Medium OR LeftProp is Medium) THEN class A.
  propBlack_Med = trapezoidal_function(features["propBlackValue"],fuzzySets[0]["Med"])
  topProp_Med = trapezoidal_function(features["topPropValue"],fuzzySets[1]["Med"])
  leftProp_Med = trapezoidal_function(features["leftPropValue"],fuzzySets[2]["Med"])
  rule1= ("A", t_norm(propBlack_Med, s_norm(topProp_Med,leftProp_Med))) #Rule Strength
  
  #Rule 2: IF PropBlack is High AND TopProp is Medium AND LeftProp is Medium THEN class B.
  propBlack_High = trapezoidal_function(features["propBlackValue"],fuzzySets[0]["High"])
  # topProp_Med and leftProp_Med calculations from Rule 1
  rule2 = ("B", t_norm(leftProp_Med, t_norm(propBlack_High,topProp_Med)))
  
  #Rule 3: IF (PropBlack is Low AND TopProp is Medium) OR LeftProp is High THEN class C.
  propBlack_Low = trapezoidal_function(features["propBlackValue"], fuzzySets[0]["Low"])
  leftProp_High = trapezoidal_function(features["leftPropValue"],fuzzySets[2]["High"])
  rule3 = ("C", s_norm(leftProp_High,t_norm(propBlack_Low,topProp_Med)))
  
  #Rule 4: IF PropBlack is Medium AND TopProp is Medium AND LeftProp is High THEN class D.
  rule4 = ("D",  t_norm(leftProp_High,t_norm(propBlack_Med, topProp_Med)))
  
  #Rule 5: IF PropBlack is High AND TopProp is Medium AND LeftProp is High THEN class E.
  rule5 = ("E", t_norm(leftProp_High, t_norm(propBlack_High,topProp_Med)))
  
  # m_combined
  class_memberships = {}
  class_memberships[rule1[0]] = rule1[1]
  class_memberships[rule2[0]] = rule2[1]
  class_memberships[rule3[0]] = rule3[1]
  class_memberships[rule4[0]] = rule4[1]
  class_memberships[rule5[0]] = rule5[1]

  
  # highest_membership_class is a string indicating the highest membership class, either "A", "B", "C", "D", or "E"
  highest_membership_class = "" 
  for key, value in class_memberships.items():
    if (value == max(class_memberships.values())):
      highest_membership_class=key

  # class_memberships is a five element list indicating the membership in each class in the order [A value, B value, C value, D value, E value]
  return highest_membership_class , list(class_memberships.values())

#---------------------Calling the functions----------------------------------------------

print('Naive Bayes Classifier: ', naive_bayes_classifier("originalExamples/Example0/input.csv"))
print('Fuzzy Classifier: ', fuzzy_classifier("originalExamples/Example0/input.csv"))





  
