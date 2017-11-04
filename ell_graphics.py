import numpy as np


def triag_facets(epoints_num, points_num):
  td = np.arange(1,points_num+1)
  I =  np.arange(1,epoints_num )
  adtime     = td[0:len(td)-1].T
  ttime_data = (adtime * epoints_num)
  adtime = (adtime-1)*epoints_num
  adtime = np.tile(adtime, (epoints_num-1,1))
  Ie = np.tile(I.T,(points_num-1,1)).T
  Ie = Ie + adtime
  Ie = Ie.flatten('F').T
  #td         = 1:1:(points_num);
  #I          = transpose(1:1:(epoints_num-1));
  #adtime     = transpose(td(1:(end-1)));
  #ttime_data = adtime*epoints_num;
  #adtime     = (adtime.'-1)*epoints_num;
  #adtime     = adtime(ones(1, epoints_num-1), :);
  #Ie         = I(:, ones(1, points_num-1)) + adtime;
  #Ie         = Ie(:);
#
  a = np.vstack((Ie, Ie+1, Ie+1+epoints_num)).T
  b = np.vstack((Ie+1+epoints_num, Ie+epoints_num, Ie)).T
  c= np.vstack((ttime_data, ttime_data+1-epoints_num, ttime_data+1)).T
  d = np.vstack((ttime_data+1, ttime_data+epoints_num, ttime_data)).T


  facets = np.zeros(((points_num - 1) * epoints_num * 2, 3))

  for i in range(0,3):

    facets[:, i] = np.hstack((a[:,i],b[:,i],c[:,i],d[:,i]))

  return facets

def main():
    epoints_num = 5
    points_num =4
    print triag_facets(epoints_num, points_num)

if __name__ == "__main__":
    main()