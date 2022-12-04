# Noise reduction using spectral reduction technique
import scipy.io.wavfile as wav
import numpy as np
import matplotlib.pyplot as plt
import os

# enter file path of sound clip
file = input("Enter the file path: ")
sr, data = wav.read(file)
#length of frame
fl = 400 
frames = []  

for i in range(0,int(len(data)/(int(fl/2))-1)):
    arr = data[int(i*int(fl/2)):int(i*int(fl/2)+fl)]  
    frames.append(arr) 

#converting the frames list into an array
frames = np.array(frames)

#using np.hamming: The window, with the maximum value normalized to one 
#(the value one appears only if the number of samples is odd)
ham_window = np.hamming(fl)
windowed_frames = frames*ham_window

#empty list containing fft of windowed_frames
dft = []
for i in windowed_frames:
     #Taking the first fourier transform (FFT) of each window
    dft.append(np.fft.fft(i))
#converting DFT into array 
dft = np.array(dft)

 #converting DFT into absolute values and finding DFT angle
dft_mag_spec = np.abs(dft)     
dft_phase_spec = np.angle(dft)  
noise_estimate = np.mean(dft_mag_spec,axis=0) 
noise_estimate_mag = np.abs(noise_estimate) 

#subtraction method and calculate the final estimate 
estimate_mag = (dft_mag_spec-2*noise_estimate_mag)
estimate_mag[estimate_mag<0]=0
estimate = estimate_mag*np.exp(1j*dft_phase_spec)

#Taking IFT as input list containing inverse fourier transform of estimate
ift = []   
for i in estimate:
    ift.append(np.fft.ifft(i))

clean_data = []
 #extending clean_data containing ift list
clean_data.extend(ift[0][:int(fl/2)])    
for i in range(len(ift)-1):   
    clean_data.extend(ift[i][int(fl/2):]+ift[i+1][:int(fl/2)])
clean_data.extend(ift[-1][int(fl/2):])   
clean_data = np.array(clean_data)


#finally plotting the graph showing the difference in the noise
fig = plt.figure(figsize=(10,7))
ax = plt.subplot(1,1,1)
ax.plot(np.linspace(0,64000,64000),data,label='Original',color="Red")
ax.plot(np.linspace(0,64000,64000),clean_data,label='Filtered',color="Blue")
ax.legend(fontsize=12)
ax.set_title('Method: Spectral Subtraction for Noise Reduction', fontsize=15)

#final filtered audio
filename = os.path.basename(file)
cleaned_file = "(Filtered_Audio)"+filename   
wav.write(cleaned_file,rate=sr, data = clean_data.astype(np.int16))

#saved file name as audio.wav(Spectral Subtraction graph).jpg
plt.savefig(filename+"(Spectral Subtraction graph).jpg")

# Noise reduction program successfully executed
print ("********************OUTPUT***********************")
print ("\nThe noise from the input audio file is reduced")