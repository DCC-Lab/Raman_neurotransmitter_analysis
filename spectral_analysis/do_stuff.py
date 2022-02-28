import spectrum
import os


my_data = {}
path = '/Users/antoinerousseau/Desktop/PM/'
for i in os.listdir(path):
    if i[0] == '.':
        continue
    if i == 'backgrounds':
        continue
    name = '{}'.format(i)
    my_data[name] = my_data.get(name, spectrum.Spectrum(path + i + '/'))

path = '/Users/antoinerousseau/Desktop/ERIKA/'
for i in os.listdir(path):
    if i[0] == '.':
        continue
    if i == 'backgrounds':
        continue
    name = '{}'.format(i)
    my_data[name] = my_data.get(name, spectrum.Spectrum(path + i + '/'))

path = '/Users/antoinerousseau/Desktop/20220124/'
for i in os.listdir(path):
    if i[0] == '.':
        continue
    if i == 'backgrounds':
        continue
    name = '{}'.format(i)
    my_data[name] = my_data.get(name, spectrum.Spectrum(path + i + '/'))

path = '/Users/antoinerousseau/Desktop/ERIKA/backgrounds/'
for i in os.listdir(path):
    if i[0] == '.':
        continue
    if i == 'backgrounds':
        continue
    name = '{}'.format(i)
    my_data[name] = my_data.get(name, spectrum.Spectrum(path + i + '/'))



path = '/Users/antoinerousseau/Desktop/20220223/'
for i in os.listdir(path):
    if i[0] == '.':
        continue
    if i == 'backgrounds':
        continue
    name = '{}'.format(i)
    my_data[name] = my_data.get(name, spectrum.Spectrum(path + i + '/'))



bg_abs = spectrum.Spectrum('/Users/antoinerousseau/Desktop/20220223/backgrounds/10min_0light/')
bg_nolight = spectrum.Spectrum('/Users/antoinerousseau/Desktop/20220223/backgrounds/10min_lightoff/')
bg_light = spectrum.Spectrum('/Users/antoinerousseau/Desktop/20220223/backgrounds/10min_lighton/')

# spectrum.Spectrum.customPlot(bg_abs.x[0], bg_nolight.y, bg_abs.y)

my_data['eau_30sec'].substract(bg_nolight.y[0], time=30)
sum_water = my_data['eau_30sec'].sum_y[0]


#big database

# my_data['cm_30sec_lightoff_1A_quartz_45uL'].substract(bg_nolight.y[0], time=30)
# my_data['cm_30sec_lightoff_1A_2'].substract(bg_nolight.y[0], time=30)
# my_data['cm_30sec_lighton_1A_quartz_45uL'].substract(bg_light.y[0], time=30)
# my_data['cm_30sec_lightoff_1B_quartz_45uL'].substract(bg_nolight.y[0], time=30)
# my_data['cm_30sec_lightoff_1B_2'].substract(bg_nolight.y[0], time=30)
# my_data['cm_30sec_lighton_1B_quartz_45uL'].substract(bg_light.y[0], time=30)
#
# my_data['cm_30sec_lightoff_2A_quartz_45uL'].substract(bg_nolight.y[0], time=30)
# my_data['cm_30sec_lightoff_2A_2'].substract(bg_nolight.y[0], time=30)
# my_data['cm_30sec_lighton_2A_quartz_45uL'].substract(bg_light.y[0], time=30)
# my_data['cm_30sec_lightoff_2B_quartz_45uL'].substract(bg_nolight.y[0], time=30)
# my_data['cm_30sec_lightoff_2B_2'].substract(bg_nolight.y[0], time=30)
# my_data['cm_30sec_lighton_2B_quartz_45uL'].substract(bg_light.y[0], time=30)
#
# my_data['cm_30sec_lightoff_3A_quartz_45uL'].substract(bg_nolight.y[0], time=30)
# my_data['cm_30sec_lightoff_3A_2'].substract(bg_nolight.y[0], time=30)
# my_data['cm_30sec_lighton_3A_quartz_45uL'].substract(bg_light.y[0], time=30)
# my_data['cm_30sec_lightoff_3B_quartz_45uL'].substract(bg_nolight.y[0], time=30)
# my_data['cm_30sec_lightoff_3B_2'].substract(bg_nolight.y[0], time=30)
# my_data['cm_30sec_lighton_3B_quartz_45uL'].substract(bg_light.y[0], time=30)
#
# my_data['cm_30sec_lightoff_4A_quartz_45uL'].substract(bg_nolight.y[0], time=30)
# my_data['cm_30sec_lightoff_4A_2'].substract(bg_nolight.y[0], time=30)
# my_data['cm_30sec_lighton_4A_quartz_45uL'].substract(bg_light.y[0], time=30)
# my_data['cm_30sec_lightoff_4B_quartz_45uL'].substract(bg_nolight.y[0], time=30)
# my_data['cm_30sec_lightoff_4B_2'].substract(bg_nolight.y[0], time=30)
# my_data['cm_30sec_lighton_4B_quartz_45uL'].substract(bg_light.y[0], time=30)
#
# my_data['cm_30sec_lightoff_1A_quartz_45uL'].substract(sum_water, bg_acq_time=300, time=30)
# my_data['cm_30sec_lightoff_1A_2'].substract(sum_water, bg_acq_time=300, time=30)
# my_data['cm_30sec_lighton_1A_quartz_45uL'].substract(sum_water, bg_acq_time=300, time=30)
# my_data['cm_30sec_lightoff_1B_quartz_45uL'].substract(sum_water, bg_acq_time=300, time=30)
# my_data['cm_30sec_lightoff_1B_2'].substract(sum_water, bg_acq_time=300, time=30)
# my_data['cm_30sec_lighton_1B_quartz_45uL'].substract(sum_water, bg_acq_time=300, time=30)
#
# my_data['cm_30sec_lightoff_2A_quartz_45uL'].substract(sum_water, bg_acq_time=300, time=30)
# my_data['cm_30sec_lightoff_2A_2'].substract(sum_water, bg_acq_time=300, time=30)
# my_data['cm_30sec_lighton_2A_quartz_45uL'].substract(sum_water, bg_acq_time=300, time=30)
# my_data['cm_30sec_lightoff_2B_quartz_45uL'].substract(sum_water, bg_acq_time=300, time=30)
# my_data['cm_30sec_lightoff_2B_2'].substract(sum_water, bg_acq_time=300, time=30)
# my_data['cm_30sec_lighton_2B_quartz_45uL'].substract(sum_water, bg_acq_time=300, time=30)
#
# my_data['cm_30sec_lightoff_3A_quartz_45uL'].substract(sum_water, bg_acq_time=300, time=30)
# my_data['cm_30sec_lightoff_3A_2'].substract(sum_water, bg_acq_time=300, time=30)
# my_data['cm_30sec_lighton_3A_quartz_45uL'].substract(sum_water, bg_acq_time=300, time=30)
# my_data['cm_30sec_lightoff_3B_quartz_45uL'].substract(sum_water, bg_acq_time=300, time=30)
# my_data['cm_30sec_lightoff_3B_2'].substract(sum_water, bg_acq_time=300, time=30)
# my_data['cm_30sec_lighton_3B_quartz_45uL'].substract(sum_water, bg_acq_time=300, time=30)
#
# my_data['cm_30sec_lightoff_4A_quartz_45uL'].substract(sum_water, bg_acq_time=300, time=30)
# my_data['cm_30sec_lightoff_4A_2'].substract(sum_water, bg_acq_time=300, time=30)
# my_data['cm_30sec_lighton_4A_quartz_45uL'].substract(sum_water, bg_acq_time=300, time=30)
# my_data['cm_30sec_lightoff_4B_quartz_45uL'].substract(sum_water, bg_acq_time=300, time=30)
# my_data['cm_30sec_lightoff_4B_2'].substract(sum_water, bg_acq_time=300, time=30)
# my_data['cm_30sec_lighton_4B_quartz_45uL'].substract(sum_water, bg_acq_time=300, time=30)
#
# #spectrum.Spectrum.customPlot(my_data['cm_30sec_lightoff_1A_quartz_45uL'].x[0], my_data['cm_30sec_lightoff_1A_quartz_45uL'].sum_y, my_data['cm_30sec_lightoff_1B_quartz_45uL'].sum_y, my_data['cm_30sec_lightoff_2A_quartz_45uL'].sum_y, my_data['cm_30sec_lightoff_2B_quartz_45uL'].sum_y, my_data['cm_30sec_lightoff_3A_quartz_45uL'].sum_y, my_data['cm_30sec_lightoff_3B_quartz_45uL'].sum_y, my_data['cm_30sec_lightoff_4A_quartz_45uL'].sum_y, my_data['cm_30sec_lightoff_4B_quartz_45uL'].sum_y)
# spectrum.Spectrum.customPlot(my_data['cm_30sec_lighton_1A_quartz_45uL'].x[0], my_data['cm_30sec_lighton_1A_quartz_45uL'].sum_y, my_data['cm_30sec_lighton_1B_quartz_45uL'].sum_y, my_data['cm_30sec_lighton_2A_quartz_45uL'].sum_y, my_data['cm_30sec_lighton_2B_quartz_45uL'].sum_y, my_data['cm_30sec_lighton_3A_quartz_45uL'].sum_y, my_data['cm_30sec_lighton_3B_quartz_45uL'].sum_y, my_data['cm_30sec_lighton_4A_quartz_45uL'].sum_y, my_data['cm_30sec_lighton_4B_quartz_45uL'].sum_y)


# ETHANOOOOOLLLL
# my_data['ethanol_10sec'].substract(bg_abs.y[0], time=10)
# my_data['ethanol_10sec'].plotSpec(WN=True)
#
# my_data['vin_1'].substract(bg_abs.y[0], time=2.5)
# my_data['vin_1'].plotSpec(sum=True, WN=True)


my_data['quartz_coverslip_10sec_peak'].substract(bg_abs.y[0], time=10)
my_data['quartz_coverslip_10sec_peak'].factor(6)
my_data['eau_distillee_30sec_1'].substract(bg_abs.y[0], time=30)
my_data['eau_distillee_30sec_1'].factor(0.64)
my_data['cm_30sec_lightoff_2A_quartz_45uL'].substract(bg_abs.y[0], time=30)
my_data['cm_30sec_lightoff_2A_2'].substract(bg_abs.y[0], time=30)
spectrum.Spectrum.customPlot(my_data['cm_30sec_lightoff_2A_2'].x[0], my_data['cm_30sec_lightoff_2A_2'].sum_y, my_data['eau_distillee_30sec_1'].sum_y, [my_data['quartz_coverslip_10sec_peak'].y[0]])

