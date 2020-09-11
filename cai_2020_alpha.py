#!/user/bin/python

#################################################################
## Cai• VCD analysis
##
## (c) Jonathan Goodman and Jonathan Lam
## 2019-2020
## jmg11@cam.ac.uk
## University of Cambridge
##
## version 1.311 June 2020
#################################################################

import sys, os
from math import exp, sqrt
from pathlib import Path
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time


###############################################################################
# Get experimental data from the output of VCD spectrometer
# Expect to have two columns of data, either tab or comma separated
# There may be a text-based heading to the file
# This ignores all lines which do not begin with a number
#
def get_data(dimension, spec_file, wav_min, wav_max):
    lst_axis = []
    with open(spec_file) as data_file:
        for line in data_file:
            if len(line.replace(","," ").split()) > 0:
                line_value = line.replace(","," ").split()
                if is_number(line_value[0].strip()):
                    if len(line_value) > 1:
                        if wav_min < float(line_value[0].strip()) < wav_max:
                            lst_axis.append(float(line_value[dimension].strip()))
    return lst_axis

###############################################################################
# Is input a number?
def is_number(isn_s):
	try:
		float(isn_s)
		return True
	except ValueError:
		return False

###############################################################################
# Get calculated data from a Schrodinger Jaguar VCD calculation of Gaussian calculation
# The data is expected to be a file called calc_file_vcd.spm
# In Gaussian, need to look for rational strengths (Rot. str.) for each IR frequency
# The Gaussian data should be in calc_files+calc_file_suffix. The form of
# calc_file_suffix is worked out elsewhere
#
def get_calc_data(calc_files, wav_min, wav_max, JaguarFile, calc_file_suffix):
    calc_data_wavenumber = []
    calc_data_intensity = []
    prepare_read_data = False
    read_data = False
    if JaguarFile:
        # The data occurs after the first occurence of the string ":::"
        # after the line "m_row". There will be several other ":::" lines
        try:
            for line in open(calc_files+"_vcd.spm"):
                if read_data:
                    if line.find(":::") < 1:
                        line_value = line.split()
                        if wav_min < float(line_value[1].strip()) < wav_max:
                            calc_data_wavenumber.append(float(line_value[1].strip()))
                            calc_data_intensity.append(float(line_value[2].strip()))
                    else:
                        read_data = False
                        prepare_read_data = False
                if line.find("m_row") > 0:
                    prepare_read_data = True
                if prepare_read_data:
                    if line.find(":::") > 0:
                        read_data = True
        except IOError as e:
            print("File open error for: ",calc_files+"_vcd.spm")
    else:
        # Gaussian File
        # The VCD data is recorded with vibrations and IR intensities
        # The strength of the VCD peaks are labelled "Rot. str."
        try:
            for line in open(calc_files+calc_file_suffix):
                if line.find("Frequencies") >= 0:
                    for i in range(2,len(line.split())):
                        calc_data_wavenumber.append(float(line.split()[i].strip()))
                if line.find("Rot. str.") >= 0:
                    for i in range(3,len(line.split())):
                        calc_data_intensity.append(float(line.split()[i].strip()))
        except IOError as e:
            print("File open error for: ",calc_files+calc_file_suffix)
    return calc_data_wavenumber, calc_data_intensity


###############################################################################
# Get energies from a Gaussian or Jaguar calculation
# These will be in calc_files+calc_file_suffix
# Also look for the DFT method, the basis set and whether
# the geometry has been optimised. These are assumed to be
# the same for all of the files
#
def get_calc_energy(calc_files, calc_file_suffix, JaguarFile):
    calc_energy = []
    optimise = "Single point"
    dft = "?"
    basis_set = "?"
    for filename in calc_files:
        check_energy_calc = len(calc_energy)
        for line in open(filename+calc_file_suffix):
            if JaguarFile:
                if line.find("Total Gibbs free energy") >= 0:
                    calc_energy.append(float(line.split()[-2]))
                elif line.find("Geometry will be optimized") >= 0:
                    optimise = "Optimized"
                elif line.find("basis set:") > 0:
                    basis_set = line.split()[2].strip()
                elif line.find("SCF calculation type") >= 0:
                    dft = line.split()[3]
            else:
                if line.find("Sum of electronic and thermal Free Energies") >= 0:
                    calc_energy.append(float(line.split()[-1]))
                elif line.find("optimizer") >= 0:
                    optimise = "Optimized"
                elif line.find("freq=VCD") > 0:
                    basis_set = line.split()[1].split("/")[1]
                    dft = line.split()[1].split("/")[0]
        if len(calc_energy) == check_energy_calc:
            # No new energy found, so add a very large number
            print("   *** Warning: No energy found in file: ",filename)
            calc_energy.append(0.000)
        # print("get_calc_energy: ",filename,calc_energy[-1])
    return calc_energy, dft, basis_set, optimise


###############################################################################
# Broaden peaks using a Lorentzian function, returning a value for each
# wavenumber in the expt_wavenums list
#
def lorentzian(expt_wavenums,calc_spectrum_wavenumbers,calc_spectrum_wnintensity,hwhm_lor_broad,scaling_factor):
    # Cycle through all (sharp) calculated peaks to find their contribution to the overall spectrum
    # and sum to find overall line-broadened spectrum
    # The intensity from a calculated peak, position p0, and position p is L:
    # L = 1/(1+x^2) where x = 2(p - p0)/w and w is hwhm_lor_broad
    # The calculated peak positions are scaled by scaling_factor before broadening
    lor_data = []
    for wavenumber in expt_wavenums:
        lor_data.append(0.0)
        for i_peak in range(0, len(calc_spectrum_wavenumbers)):
            x_value = 2.0*(wavenumber-calc_spectrum_wavenumbers[i_peak]*scaling_factor)/hwhm_lor_broad
            lor_data[-1] += calc_spectrum_wnintensity[i_peak] / (1 + x_value*x_value)
    return lor_data

###############################################################################
# match_score = match_score_calc(calc_signals,alfa_expt_signals)
# Find the products of the calculated and experimental signal at each point
# Return both the normalised sum of products and the total product
# The former should be between -1 and 1; the latter may give a sense of the strength of the signal
#
def match_score_calc(calc_signals, expt_signals):
    sum_alfacalc = 0.0
    sum_calc2 = 0.0
    sum_a2 = 0.0
    for i in range(0, len(calc_signals)):
        sum_alfacalc += calc_signals[i]*expt_signals[i]
        sum_calc2 += calc_signals[i]*calc_signals[i]
        sum_a2 += expt_signals[i]*expt_signals[i]
    return [sum_alfacalc/sqrt(sum_calc2*sum_a2), sum_alfacalc]

###############################################################################
# Are the results in Jaguar or Gaussian files?
#
def jaguar_or_gaussian(calculation_filename):
    JaguarFile = False
    GaussianFile = False
    calc_file_suffix = ".out"
    test_file_name = calculation_filename+calc_file_suffix
    if Path(test_file_name).is_file():
        with open(test_file_name, 'r') as calc_data_file:
            for input_file_line in calc_data_file:
                if input_file_line.find("Jaguar") >= 0:
                    JaguarFile = True
                    break
                elif input_file_line.find("Gaussian") >= 0:
                    GaussianFile = True
                    break
    else:
        calc_file_suffix = ".log"
        test_file_name = calculation_filename+calc_file_suffix
        if Path(test_file_name).is_file():
            with open(test_file_name, 'r') as calc_data_file:
                for input_file_line in calc_data_file:
                    if input_file_line.find("Gaussian") >= 0:
                        GaussianFile = True
                        break
        else:
            calc_file_suffix = ""
            test_file_name = calculation_filename
            if Path(test_file_name).is_file():
                with open(test_file_name, 'r') as calc_data_file:
                    for input_file_line in calc_data_file:
                        if input_file_line.find("Gaussian") >= 0:
                            GaussianFile = True
                            break
    # If JaguarFile is True, then expect data in calc_files[].out
    # If GaussianFile is True, then expect data in calc_files[].out or .log
    # If neither are true, check calc_files[] and calc_files[].log
    return [JaguarFile, GaussianFile, calc_file_suffix]

###############################################################################
# Print out results
#
def printout_confidence(calc_label, sf_label, preferred_scaling_factor, comparison, max_confidence):
    if calc_label.find("File (B)") >= 0:
        # print("invert for File (B)")
        comparison = -comparison
    if comparison > 0:
        conclusion = "File (A) is assigned to the enantiomer calculated"
        if abs(max_confidence) < insufficient_information_criterion:
            conclusion = "Insufficient information for a definite conclusion (A)"
    else:
        conclusion = "File (A) is NOT the enantiomer calculated"
        if abs(max_confidence) < insufficient_information_criterion:
            conclusion = "Insufficient information for a definite conclusion (B)"
    print(calc_label, sf_label, ": {:6.3f}".format(preferred_scaling_factor),
        "; ",conclusion,"with Cai.factor{:3.0f}".format(abs(max_confidence)))
    print(calc_label, sf_label, ": {:6.3f}".format(preferred_scaling_factor),
        "; ",conclusion,"with Cai.factor{:3.0f}".format(abs(max_confidence)), file=logfile)
    return

###############################################################################
# Print out two line output .csv results
#
def print_tlo_summary(one_line_output_decode, one_line_output):
    two_line_output_file = open(sys.argv[1].split(".")[0]+"_tlo.csv", 'w')
    # print(one_line_output_decode)
    # print(one_line_output)
    print_string = ','.join(map(str, one_line_output_decode))
    print(print_string, file=two_line_output_file)
    print_string = ','.join(map(str, one_line_output))
    print(print_string, file=two_line_output_file)
    two_line_output_file.close()
    return



###############################################################################
# Print out a graph
#
def print_graph_files(total_confidence,goodness_description):
    fig, (ax1, ax2) = plt.subplots(2,1)  # Create a figure containing two axes.
    if reverse_xaxis:
        ax1.invert_xaxis()
        ax2.invert_xaxis()
    max_alfa=0.0
    max_combine=0.0
    max_blank_correct=0.0
    max_calc=0.0
    for i in range(0,len(alfa_expt_signals)):
        if max_alfa < abs(alfa_expt_signals[i]):
            max_alfa = abs(alfa_expt_signals[i])
        if max_calc < abs(calc_signals[i]):
            max_calc = abs(calc_signals[i])
        if single_enantiomer:
            if blank_file_marker:
                if max_blank_correct < abs(alfa_expt_signals_blank_correction[i]):
                    max_blank_correct = abs(alfa_expt_signals_blank_correction[i])
        else:
            if max_combine < abs(corr_alfa_expt_signals[i]):
                max_combine = abs(corr_alfa_expt_signals[i])
    calc_scale_factor = max_calc/max_alfa
    if single_enantiomer:
        if blank_file_marker:
            calc_scale_factor = max_calc/max_blank_correct
    else:
        calc_scale_factor = max_calc/max_combine
    scale_calc_signals = []
    for i in range(0,len(calc_signals)):
        scale_calc_signals.append(calc_signals[i]/calc_scale_factor)
    if total_confidence > 0:
        ax1.set_title('(A) favoured - Cai.factor '+str(int(total_confidence))+' ('+goodness_description+')\n'+sys.argv[1].split(".")[0])
    else:
        ax1.set_title('(B) favoured - Cai.factor '+str(int(-total_confidence))+' ('+goodness_description+')\n'+sys.argv[1].split(".")[0])
    ax1.plot(expt_wavenums, alfa_expt_signals, label='expt (A)', color='blue')
    if not single_enantiomer:
        ax1.plot(expt_wavenums, beta_expt_signals, label='expt (B)', color='orange')
    if blank_file_marker:
        # blank_file_intensity
        ax1.plot(expt_wavenums, blank_file_intensity, label='blank', color='black')
        # ax1.plot(blank_file_wavenumber_raw, blank_file_signal_raw, label='raw blank')
    ax1.legend()
    if single_enantiomer:
        if blank_file_marker:
            ax2.plot(expt_wavenums, alfa_expt_signals_blank_correction, label='blank correction', color='cyan')
    else:
        ax2.plot(expt_wavenums, corr_alfa_expt_signals, label='combined expt', color='magenta')
    ax2.plot(expt_wavenums, scale_calc_signals, label='calc. (A)', color='green')
    ax2.set_xlabel('wavenumbers')
    ax2.legend()
    # print("Printing figure",sys.argv[1].split(".")[0]+'.pdf')
    fig.savefig(sys.argv[1].split(".")[0]+'.pdf')
    plt.close(fig)
    return
###############################################################################






###############################################################################
# Summarise conclusions
#
def summarise_conclusion():
    # print("**** summarise_conclusion ****")
    # combined cai = bsf_confidence[3] (A=[0]; B=[1])
    # opt SF values = max_confidence[3] (A=[0]; B=[1]) with SF preferred_scaling_factor[3],etc
    # average is average_confidence[3] with range max_average_confidence[3]-min_average_confidence[3]
    # If blank_file_marker: also do 4 and 5
    # overall confidence = highest of bsf_confidence 3, 0, 1, 4, 5
    # plus more if both 0/1 or 4/5 are high
    # plus more if max_confidence is higher and preferred_scaling_factor is OK
    # plus more if average_confidence is good too
    # remember to invert for B
    # goodness_range=[10.0,20.0,30.0,40.0,50.0]
    # goodness_descriptors for range 0-9.9, 10-19.9, 20-29.9, 30-39.9, >40.0
    # goodness_descriptors=["uncertain","possible","cautious confidence","confident","very confident","highly confident","supremely confident","confident (4)","confident (5)","confident (6)"]
    goodness_descriptors=["uncertain","possible","cautiously confident","fairly confident","confident","very confident","very confident","very confident","very confident","very confident","mixed"]
    total_confidence=0.0

    if single_enantiomer:
        print("Single enantiomer summary")
        print("Single enantiomer summary", file=logfile)
        total_confidence=bsf_confidence[0]
        if total_confidence > 0:
            assign_to_file_a=1.0
            confidence_string="that File (A) is the enantiomer calculated"
        else:
            assign_to_file_a=-1.0
            confidence_string="that File (B) is the enantiomer calculated"
        total_confidence_measure = 0
        print("This is based on File (A) (",goodness_descriptors[int(abs(bsf_confidence[0]*assign_to_file_a/10.0))],")")
        print("This is based on File (A) (",goodness_descriptors[int(abs(bsf_confidence[0]*assign_to_file_a/10.0))],")", file=logfile)
        if blank_file_marker:
            print("Using the blank data, File (A) is ",goodness_descriptors[int(abs(bsf_confidence[4]*assign_to_file_a/10.0))])
            print("Using the blank data, File (A) is ",goodness_descriptors[int(abs(bsf_confidence[4]*assign_to_file_a/10.0))], file=logfile)
        if abs(total_confidence) < abs(bsf_confidence[4]):
            total_confidence = bsf_confidence[4]
            total_confidence_measure = 4
        old_total_confidence = total_confidence
        scale_total_confidence = []
        x_value = 2.0*(defined_scaling_factor-preferred_scaling_factor[0])/extreme_sf_warning
        lor_data = 1 / (1 + x_value*x_value)
        scale_total_confidence.append(bsf_confidence[0]*(1-lor_data)+max_confidence[0]*lor_data)
        if abs(total_confidence) < abs(scale_total_confidence[0]):
            total_confidence = scale_total_confidence[0]
            total_confidence_measure = 6
        if blank_file_marker:
            x_value = 2.0*(defined_scaling_factor-preferred_scaling_factor[4])/extreme_sf_warning
            lor_data = 1 / (1 + x_value*x_value)
            scale_total_confidence.append(bsf_confidence[4]*(1-lor_data)+max_confidence[4]*lor_data)
            if abs(total_confidence) < abs(scale_total_confidence[1]):
                total_confidence = scale_total_confidence[1]
                total_confidence_measure = 10
            if blank_precedence:
                total_confidence = scale_total_confidence[1]
                total_confidence_measure = 10
        if abs(total_confidence)-abs(old_total_confidence) >= 1:
            print("Optimising the scale factor increases the confidence level: ",goodness_descriptors[int(abs(total_confidence*assign_to_file_a/10.0))])
            print("Optimising the scale factor increases the confidence level: ",goodness_descriptors[int(abs(total_confidence*assign_to_file_a/10.0))], file=logfile)
        else:
            print("Optimising the scale factor does not improve overall confidence substantially")
            print("Optimising the scale factor does not improve overall confidence substantially", file=logfile)
            # print(lor_data,total_confidence)
        x_value = 2.0*(defined_scaling_factor-preferred_scaling_factor[0])/extreme_sf_warning
        lor_data = 1 / (1 + x_value*x_value)
        if abs(int((max_confidence[0]-bsf_confidence[0])*lor_data/10.0)) >= 1:
            print("File (A) assignment improved to",goodness_descriptors[int(abs((bsf_confidence[0]*(1-lor_data)+max_confidence[0]*lor_data)/10.0))],"by scale factor",preferred_scaling_factor[0])
            print("File (A) assignment improved to",goodness_descriptors[int(abs((bsf_confidence[0]*(1-lor_data)+max_confidence[0]*lor_data)/10.0))],"by scale factor",preferred_scaling_factor[0], file=logfile)
    else:
        # print(int(bsf_confidence[3]),int(bsf_confidence[0]),int(bsf_confidence[1]))
        # print(int(max_confidence[3]),int(max_confidence[0]),int(max_confidence[1]))
        # print(preferred_scaling_factor[3],preferred_scaling_factor[0],preferred_scaling_factor[1])
        # print(int(average_confidence[3]),int(average_confidence[0]),int(average_confidence[1]))
        # print(int(max_average_confidence[3]),int(max_average_confidence[0]),int(max_average_confidence[1]))
        # print(int(min_average_confidence[3]),int(min_average_confidence[0]),int(min_average_confidence[1]))
        # print(int(max_average_confidence[3]-min_average_confidence[3]),int(max_average_confidence[0]-min_average_confidence[0]),int(max_average_confidence[1]-min_average_confidence[1]))
        print("Summary:")
        print("Summary:", file=logfile)
        total_confidence=bsf_confidence[3]
        total_confidence_measure = 3
        if bsf_confidence[3] > 0:
            assign_to_file_a=1.0
            confidence_string="that File (A) is the enantiomer calculated"
        else:
            assign_to_file_a=-1.0
            confidence_string="that File (B) is the enantiomer calculated"
        print("Based on combined spectra,",goodness_descriptors[int(abs(total_confidence*assign_to_file_a/10.0))],confidence_string)
        print("This is based on File (A) (",goodness_descriptors[int(abs(bsf_confidence[0]*assign_to_file_a/10.0))],"), and File (B) (",goodness_descriptors[abs(int(bsf_confidence[1]*assign_to_file_a/10.0))],")")
        if blank_file_marker:
            print("Using the blank data, File (A) is ",goodness_descriptors[int(abs(bsf_confidence[4]*assign_to_file_a/10.0))],", and File (B) is",goodness_descriptors[abs(-1*int(bsf_confidence[5]*assign_to_file_a/10.0))])
            print("Using the blank data, File (A) is ",goodness_descriptors[int(abs(bsf_confidence[4]*assign_to_file_a/10.0))],", and File (B) is",goodness_descriptors[abs(-1*int(bsf_confidence[5]*assign_to_file_a/10.0))], file=logfile)
        if abs(total_confidence) < abs(bsf_confidence[0]):
            total_confidence = bsf_confidence[0]
            total_confidence_measure = 0
        if abs(total_confidence) < abs(bsf_confidence[1]):
            total_confidence = -bsf_confidence[1]
            total_confidence_measure = 1
        if abs(total_confidence) < abs(bsf_confidence[2]):
            total_confidence = bsf_confidence[2]
            total_confidence_measure = 2
        if blank_file_marker:
            if abs(total_confidence) < abs(bsf_confidence[4]):
                total_confidence = bsf_confidence[4]
                total_confidence_measure = 4
            if abs(total_confidence) < abs(bsf_confidence[5]):
                total_confidence = -bsf_confidence[5]
                total_confidence_measure = 5
            if blank_precedence:
                if abs(bsf_confidence[4]) < abs(bsf_confidence[5]):
                    total_confidence = -bsf_confidence[5]
                else:
                    total_confidence = bsf_confidence[4]
        # print("Non-optimised confidence ",total_confidence,total_confidence_measure)
        # Do A and B disagree?
        # Perhaps should reduce total confidence if this happens? By how much?
        # if bsf_confidence[0]*bsf_confidence[3] < 0:
        #     print("File (A) disagrees with overall analysis")
        # elif bsf_confidence[1]*bsf_confidence[3] > 0:
        #     print("File (B) disagrees with overall analysis")
        # The optimised scale factor is only trusted if it is close to the defined_scaling_factor
        # A Lorentzian is used to average the results of the defined and unoptimised values
        # If the optimised value is similar to the defined one, it is combined
        # If it is different, it is ignored
        # Lorentzian:
        # The intensity from a defined_scaling_factor, position p0, and position p is L:
        # L = 1/(1+x^2) where x = 2(p - p0)/w and w is extreme_sf_warning
        scale_total_confidence = []
        x_value = 2.0*(defined_scaling_factor-preferred_scaling_factor[0])/extreme_sf_warning
        lor_data = 1 / (1 + x_value*x_value)
        scale_total_confidence.append(bsf_confidence[0]*(1-lor_data)+max_confidence[0]*lor_data)
        x_value = 2.0*(defined_scaling_factor-preferred_scaling_factor[1])/extreme_sf_warning
        lor_data = 1 / (1 + x_value*x_value)
        scale_total_confidence.append(bsf_confidence[1]*(1-lor_data)+max_confidence[1]*lor_data)
        x_value = 2.0*(defined_scaling_factor-preferred_scaling_factor[2])/extreme_sf_warning
        lor_data = 1 / (1 + x_value*x_value)
        scale_total_confidence.append(bsf_confidence[2]*(1-lor_data)+max_confidence[2]*lor_data)
        x_value = 2.0*(defined_scaling_factor-preferred_scaling_factor[3])/extreme_sf_warning
        lor_data = 1 / (1 + x_value*x_value)
        scale_total_confidence.append(bsf_confidence[3]*(1-lor_data)+max_confidence[3]*lor_data)
        #print("scaling test ",x_value,lor_data,total_confidence,scale_total_confidence[3],total_confidence)
        #print(total_confidence,scale_total_confidence)
        old_total_confidence = total_confidence
        if abs(total_confidence) < abs(scale_total_confidence[0]):
            total_confidence = scale_total_confidence[0]
            total_confidence_measure = 6
        if abs(total_confidence) < abs(scale_total_confidence[1]):
            total_confidence = -scale_total_confidence[1]
            total_confidence_measure = 7
        if abs(total_confidence) < abs(scale_total_confidence[2]):
            total_confidence = scale_total_confidence[2]
            total_confidence_measure = 8
        if abs(total_confidence) < abs(scale_total_confidence[3]):
            total_confidence = scale_total_confidence[3]
            total_confidence_measure = 9
        if blank_file_marker:
            x_value = 2.0*(defined_scaling_factor-preferred_scaling_factor[4])/extreme_sf_warning
            lor_data = 1 / (1 + x_value*x_value)
            scale_total_confidence.append(bsf_confidence[4]*(1-lor_data)+max_confidence[4]*lor_data)
            x_value = 2.0*(defined_scaling_factor-preferred_scaling_factor[5])/extreme_sf_warning
            lor_data = 1 / (1 + x_value*x_value)
            scale_total_confidence.append(bsf_confidence[5]*(1-lor_data)+max_confidence[5]*lor_data)
            if abs(total_confidence) < abs(scale_total_confidence[4]):
                total_confidence = scale_total_confidence[4]
                total_confidence_measure = 10
            if abs(total_confidence) < abs(scale_total_confidence[5]):
                total_confidence = -scale_total_confidence[5]
                total_confidence_measure = 11
            if blank_precedence:
                if total_confidence_measure == 4:
                    if abs(bsf_confidence[4]) < abs(scale_total_confidence[4]):
                        total_confidence = scale_total_confidence[4]
                elif total_confidence_measure == 5:
                    if abs(bsf_confidence[5]) < abs(scale_total_confidence[5]):
                        total_confidence = -scale_total_confidence[5]

                # print("Blank B corrected ",scale_total_confidence,total_confidence)
        #print("test** total confidence",total_confidence,assign_to_file_a,int(abs(total_confidence*assign_to_file_a/10.0)))
        if abs(total_confidence)-abs(old_total_confidence) >= 1:
            print("Optimising the scale factor increases the confidence level to: ",goodness_descriptors[int(abs(total_confidence*assign_to_file_a/10.0))])
            print("Optimising the scale factor increases the confidence level to: ",goodness_descriptors[int(abs(total_confidence*assign_to_file_a/10.0))], file=logfile)
        else:
            print("Optimising the scale factor does not improve overall confidence substantially")
            print("Optimising the scale factor does not improve overall confidence substantially", file=logfile)
            # print(lor_data,total_confidence)
        x_value = 2.0*(defined_scaling_factor-preferred_scaling_factor[0])/extreme_sf_warning
        lor_data = 1 / (1 + x_value*x_value)
        if abs(int((max_confidence[0]-bsf_confidence[0])*lor_data/10.0)) >= 1:
            print("File (A) assignment improved to",goodness_descriptors[int(abs((bsf_confidence[0]*(1-lor_data)+max_confidence[0]*lor_data)/10.0))],"by scale factor",preferred_scaling_factor[0])
            print("File (A) assignment improved to",goodness_descriptors[int(abs((bsf_confidence[0]*(1-lor_data)+max_confidence[0]*lor_data)/10.0))],"by scale factor",preferred_scaling_factor[0], file=logfile)
        x_value = 2.0*(defined_scaling_factor-preferred_scaling_factor[1])/extreme_sf_warning
        lor_data = 1 / (1 + x_value*x_value)
        if abs(int((-max_confidence[1]+bsf_confidence[1])*lor_data/10.0)) >= 1:
            print("File (B) assignment improved to",goodness_descriptors[int(abs(-1.0*(bsf_confidence[1]*(1-lor_data)+max_confidence[1]*lor_data)/10.0))],"by scale factor",preferred_scaling_factor[1])
            print("File (B) assignment improved to",goodness_descriptors[int(abs(-1.0*(bsf_confidence[1]*(1-lor_data)+max_confidence[1]*lor_data)/10.0))],"by scale factor",preferred_scaling_factor[1], file=logfile)
        # if blank_file_marker:
        #     # This is extra information, so should be able to increase the confidence
        #     # The correction is slightly arbitrary
        #     if bsf_confidence[4]*assign_to_file_a > bsf_confidence[0]*assign_to_file_a:
        #         print("Allowing for the blank data, File (A) increases the confidence in its result (",goodness_descriptors[int(bsf_confidence[4]*assign_to_file_a/10.0)],")")
        #         total_confidence += (bsf_confidence[4]-bsf_confidence[0])/2.0
        #     if bsf_confidence[5]*assign_to_file_a < bsf_confidence[1]*assign_to_file_a:
        #         print("Allowing for the blank data, File (B) increases the confidence in its result (",goodness_descriptors[int(-1.0*bsf_confidence[5]*assign_to_file_a/10.0)],")")
        #         total_confidence -= (bsf_confidence[5]-bsf_confidence[1])/2.0

    print()
    print("Overall Cai.factor is",int(total_confidence),"which means",goodness_descriptors[int(abs(total_confidence*assign_to_file_a/10.0))],"assignment")
    print("Overall Cai.factor is",int(total_confidence),"which means",goodness_descriptors[int(abs(total_confidence*assign_to_file_a/10.0))],"assignment", file=logfile)
    print("Based on total confidence measure ",total_confidence_measure, file=logfile)
    one_line_output_decode.append("Overall Cai.Factor")
    one_line_output.append(total_confidence)
    one_line_output_decode.append("total confidence measure")
    one_line_output.append(total_confidence_measure)

    print()
    return total_confidence, goodness_descriptors[int(abs(total_confidence*assign_to_file_a/10.0))]


################################################################################
################################################################################
################################################################################
##########################         Start Cai•         ##########################
################################################################################
################################################################################
################################################################################

logfile = open(sys.argv[1].split(".")[0]+".log", 'w')
output_string = (
"##########################################\n"
+"##          Cai• VCD analysis           ##\n"
+"##    University of Cambridge, 2020     ##\n"
+"##########################################\n")
print(output_string)
print(output_string, file=logfile)
print(sys.argv[1]+"\n")
print(sys.argv[1]+"\n", file=logfile)

# Default parameters
# These can be altered in the input files
molecule_title=sys.argv[1].split(".")[0]
minimum_wavenumber = 1000.0
maximum_wavenumber = 1600.0
hwhm_lor_broad = 5.0
temperature = 300.0
R_in_hartrees = 0.0000031668
defined_scaling_factor = 0.975
# Give a warning message if the optimised scale factor is too far from the defined one
# Warning if optimised is more than extreme_sf_warning from defined
extreme_sf_warning = 0.01
# Duplicate structure removal
# If the difference in energy of two structures is less than
# max_energy_difference, they are investigated for similarity
# Value in Hartrees. 0.0001 Hartrees is about 0.3 kJ/mol
max_energy_difference = 0.0001
# Calculated vcd peaks are considered equivalent if they match
# withing max_vcd_frequency_difference wavenumbers
max_vcd_frequency_difference = 2.0
# By default, all distinct input structures are used in the analysis
# If boltzmann_analysis = False, only the lowest energy one is used
boltzmann_analysis = True
# ignore all structures with energies more than boltzmann_cutoff (kJ/mol) above the global minimum
boltzmann_cutoff = 10.0
# The value of confidence required to suggest an assignment
insufficient_information_criterion = 10
# Print graphs of data and calculations
print_graph = True
# Should wavenumbers increase or decrease left to right?
# Default is now to decrease
reverse_xaxis = True
# Give blank correction precedence
blank_precedence = True



###############################################################################
# Read in data from the command file
single_enantiomer = True
expt_files_marker = False
calc_files_marker = False
blank_file_marker = False
settings_marker = False
print_spectra_csv = False
print_scaling_factor_csv = False
print_summary_csv = False
calc_files = []
beta_expt_file = ""
blank_file = ""
for input_file_line in open(sys.argv[1]):
    # print(input_file_line,calc_files_marker,expt_files_marker,single_enantiomer)
    if len(input_file_line) > 1:
        if input_file_line.find("#") != 0:
            if input_file_line.lower().find("<settings>") >= 0:
                blank_file_marker = False
                expt_files_marker = False
                calc_files_marker = False
                settings_marker = True
            elif input_file_line.lower().find("<experiments>") >= 0:
                blank_file_marker = False
                expt_files_marker = True
                calc_files_marker = False
                settings_marker = False
            elif input_file_line.lower().find("<calculations>") >= 0:
                blank_file_marker = False
                expt_files_marker = False
                calc_files_marker = True
                settings_marker = False
            elif input_file_line.lower().find("<blank>") >= 0:
                blank_file_marker = True
                expt_files_marker = False
                calc_files_marker = False
                settings_marker = False
            elif input_file_line.lower().find("<") >= 0:
                blank_file_marker = False
                expt_files_marker = False
                calc_files_marker = False
                settings_marker = False
            elif expt_files_marker:
                if single_enantiomer:
                    alfa_expt_file = input_file_line.strip()
                    single_enantiomer = False
                else:
                    beta_expt_file = input_file_line.strip()
                    expt_files_marker = False
            elif calc_files_marker:
                calc_files.append(input_file_line.strip())
            elif blank_file_marker:
                blank_file = input_file_line.strip()
            elif settings_marker:
                if input_file_line.lower().find("title:") >= 0:
                    molecule_title = input_file_line.split()[1]
                elif input_file_line.lower().find("broadening") >= 0:
                    hwhm_lor_broad = float(input_file_line.split()[1])
                elif input_file_line.lower().find("minimum_wavenumber") >= 0:
                    minimum_wavenumber = float(input_file_line.split()[1])
                elif input_file_line.lower().find("maximum_wavenumber") >= 0:
                    maximum_wavenumber = float(input_file_line.split()[1])
                elif input_file_line.lower().find("temperature") >= 0:
                    temperature = float(input_file_line.split()[1])
                elif input_file_line.lower().find("defined_scaling_factor") >= 0:
                    defined_scaling_factor = float(input_file_line.split()[1])
                elif input_file_line.lower().find("boltzmann_analysis") >= 0:
                    if input_file_line.lower().split()[1] == "false":
                        boltzmann_analysis = False
                elif input_file_line.lower().find("print_graph") >= 0:
                    if input_file_line.lower().split()[1] == "false":
                        print_graph = False
                elif input_file_line.lower().find("reverse_xaxis") >= 0:
                    if input_file_line.lower().split()[1] == "false":
                        reverse_xaxis = False
                elif input_file_line.lower().find("boltzmann_cutoff") >= 0:
                    boltzmann_cutoff = float(input_file_line.split()[1])
                elif input_file_line.lower().find("extreme_sf_warning") >= 0:
                    extreme_sf_warning = float(input_file_line.split()[1])
                elif input_file_line.lower().find("max_unique_energy_difference") >= 0:
                    max_energy_difference = float(input_file_line.split()[1])
                elif input_file_line.lower().find("max_unique_vcd_frequency_difference") >= 0:
                    max_vcd_frequency_difference = float(input_file_line.split()[1])
                elif input_file_line.lower().find("blank_precedence") >= 0:
                    if input_file_line.lower().split()[1] == "false":
                        blank_precedence = False
                elif input_file_line.lower().find("print_csv") >= 0:
                    if input_file_line.lower().find("spectra") >= 0:
                        print_spectra_csv = True
                    if input_file_line.lower().find("scaling_factor") >= 0:
                        print_scaling_factor_csv = True
                    if input_file_line.lower().find("summary") >= 0:
                        print_summary_csv = True
single_enantiomer = True
if len(beta_expt_file) > 0:
    single_enantiomer = False
blank_file_marker = False
if len(blank_file) > 0:
    blank_file_marker = True
boltzmann_cutoff_hartree = boltzmann_cutoff/2625.5

###############################################################################
# print the input information from the command file
print("minimum wavenumber: ",minimum_wavenumber)
print("minimum wavenumber: ",minimum_wavenumber, file=logfile)
print("maximum wavenumber: ",maximum_wavenumber)
print("maximum wavenumber: ",maximum_wavenumber, file=logfile)
print("Lorentzian broadening: ",hwhm_lor_broad)
print("Lorentzian broadening: ",hwhm_lor_broad, file=logfile)
print("Temperature for Boltzmann averaging: ",temperature," K")
print("Temperature for Boltzmann averaging (Kelvin): ",temperature, file=logfile)
print("Defined scaling factor: ", defined_scaling_factor)
print("Defined scaling factor: ", defined_scaling_factor, file=logfile)
print("Print out graphs: ", print_graph)
print("Print out graphs: ", print_graph, file=logfile)
print("Boltzmann Analysis: ", boltzmann_analysis)
print("Boltzmann Analysis: ", boltzmann_analysis, file=logfile)
print("Boltzmann Analysis Energy cut-off: ", boltzmann_cutoff)
print("Boltzmann Analysis Energy cut-off: ", boltzmann_cutoff, file=logfile)
print("Unique calculated structure criteria: Energy: ", max_energy_difference,"Frequency: ", max_vcd_frequency_difference)
print("Unique calculated structure criteria: Energy: ", max_energy_difference,"Frequency: ", max_vcd_frequency_difference, file=logfile)
print("Extreme scale factor warning range: ", extreme_sf_warning)
print("Extreme scale factor warning range: ", extreme_sf_warning, file=logfile)
print("Insufficient information criterion for Cai.factor: ", insufficient_information_criterion)
print("Insufficient information criterion for Cai.factor: ", insufficient_information_criterion, file=logfile)

if print_spectra_csv:
    print("Printing spectra in .csv file")
    print("Printing spectra in .csv file", file=logfile)
if print_scaling_factor_csv:
    print("Printing scaling factor analysis in .csv file")
    print("Printing scaling factor analysis in .csv file", file=logfile)
if print_summary_csv:
    print("Printing two-line summary in .csv file")
    print("Printing two-line summary in .csv file", file=logfile)
print("")
print("", file=logfile)
if single_enantiomer:
    # print("Single enantiomer: ",alfa_expt_file)
    print("Experimental data for single enantiomer:\n(A) ",alfa_expt_file)
    print("Experimental data for single enantiomer:\n(A) ",alfa_expt_file, file=logfile)
else:
    print("Experimental data for both enantiomers:\n(A) ",alfa_expt_file,"\n(B) ",beta_expt_file)
    print("Experimental data for both enantiomers:\n(A) ",alfa_expt_file,"\n(B) ",beta_expt_file, file=logfile)
if blank_file_marker:
    print("Blank file:\n",blank_file)
    print("Blank file:\n",blank_file, file=logfile)

print("\nCalculation files:")
print('\n'.join(calc_files))
print()
print("\nCalculation files:", file=logfile)
print('\n'.join(calc_files), file=logfile)
print("\n", file=logfile)

###############################################################################
# Read in the experimental data
# if single_enantiomer; beta_expt_file assigned as the reverse of alfa_expt_file
expt_wavenums = get_data(0, alfa_expt_file, minimum_wavenumber, maximum_wavenumber)
alfa_expt_signals = get_data(1, alfa_expt_file, minimum_wavenumber, maximum_wavenumber)
if single_enantiomer:
    beta_expt_signals = []
    for i in alfa_expt_signals:
        beta_expt_signals.append(-i)
else:
    beta_expt_signals = get_data(1, beta_expt_file, minimum_wavenumber, maximum_wavenumber)
    # check both experimental files have same wavenumber scale
    if expt_wavenums != get_data(0, beta_expt_file, minimum_wavenumber, maximum_wavenumber):
        print('ERROR: Different wavenumber scale for experimental spectra! Exiting.')
        exit()

one_line_output_decode = ["name", "Molecule name", "Spectrometer"]
one_line_output = [sys.argv[1], molecule_title]
if alfa_expt_file.find("VCD.PRN") >= 0:
    one_line_output.append("Biotools")
elif alfa_expt_file.find(".dpt") >= 0:
    one_line_output.append("Bruker")
else:
    one_line_output.append("Unknown")

###############################################################################
#subtractive baseline correction: generates corr_expt_signals
corr_alfa_expt_signals = []
check_corr_size = 0.0
for i in range (0, len(alfa_expt_signals)):
    corr_alfa_expt_signals.append((alfa_expt_signals[i]-beta_expt_signals[i])/2.0)
    check_corr_size += corr_alfa_expt_signals[-1]*corr_alfa_expt_signals[-1]
if check_corr_size < 0.0000000001:
    print("Two experimental spectra are almost identical! Exiting.")
    print("Two experimental spectra are almost identical! Exiting.", file=logfile)
    exit()

###############################################################################
# Blank file handling
alfa_expt_signals_blank_correction = []
beta_expt_signals_blank_correction = []
if blank_file_marker:
    blank_file_wavenumber_raw = get_data(0, blank_file, minimum_wavenumber, maximum_wavenumber)
    blank_file_signal_raw = get_data(1, blank_file, minimum_wavenumber, maximum_wavenumber)
    # print(blank_file_signal_raw)
    if blank_file_wavenumber_raw == expt_wavenums:
        blank_file_intensity = blank_file_signal_raw
        # print("blank_file_wavenumber_raw == expt_wavenums")
    else:
        # need to scale blank_file to fit other experimental data
        # Use linear interpolation
        # Does direction match?
        blank_length = len(blank_file_wavenumber_raw)
        bdirection = 1
        bposition = 0
        if blank_file_wavenumber_raw[0] > blank_file_wavenumber_raw[-1]:
            blank_file_wavenumber_raw.reverse()
            blank_file_signal_raw.reverse()
        # Blank and new data go in the same direction as each other
        nearest_smaller_peak = 0
        blank_file_intensity = []
        # Now create a new file with a blank measurement for each of the experimental wavenumbers
        for i in range(0, len(expt_wavenums)):
            # print(expt_wavenums[i],len(blank_file_wavenumber_raw),blank_file_wavenumber_raw[0],blank_file_wavenumber_raw[-1])
            if (expt_wavenums[i] < blank_file_wavenumber_raw[0]):
                nearest_smaller_peak = 0
            else:
                # print([l for l in blank_file_wavenumber_raw if l<expt_wavenums[i]][-1])
                nearest_smaller_peak = blank_file_wavenumber_raw.index([l for l in blank_file_wavenumber_raw if l<expt_wavenums[i]][-1])
            # num = List.index([l for l in List if l<maxNum][-1])
            gap1 = abs(expt_wavenums[i]-blank_file_wavenumber_raw[nearest_smaller_peak])
            if nearest_smaller_peak < len(blank_file_wavenumber_raw):
                gap2 = abs(expt_wavenums[i]-blank_file_wavenumber_raw[nearest_smaller_peak+1])
            else:
                gap2 = gap1
            new_intensity = (blank_file_signal_raw[nearest_smaller_peak]*gap1 + blank_file_signal_raw[nearest_smaller_peak+1]*gap2)/(gap1+gap2)
            blank_file_intensity.append(new_intensity)
            # print(i, len(expt_wavenums), len(blank_file_wavenumber_raw), expt_wavenums[i], nearest_smaller_peak, new_intensity)
else:
    blank_file_intensity = [0.0]*len(expt_wavenums)
for i in range(0,len(expt_wavenums)):
    alfa_expt_signals_blank_correction.append(alfa_expt_signals[i] - blank_file_intensity[i])
    beta_expt_signals_blank_correction.append(beta_expt_signals[i] - blank_file_intensity[i])
    # print(blank_file_intensity[i])


###############################################################################
# Now need to import data from Jaguar or Gaussian, add Lorentzian line broadening,
# and create compatible signal for comparison with experiment
# Calculated files may be given as a list of files or as a directory
# Is calc_files a list of files or a directory?
#
if len(calc_files) == 1:
    print("Only one filename listed for calculations")
    print("Only one filename listed for calculations", file=logfile)
    if Path(calc_files[0]).is_dir():
        print("which is a directory")
        print("which is a directory containing the following files:", file=logfile)
        # replace calc_files with a list of .out files in the directory
        # For Gaussian calculations, just use .log
        # For Jaguar, use .out
        path_list = list(Path(calc_files[0]).glob('*.log'))
        JaguarFile, GaussianFile, calc_file_suffix = jaguar_or_gaussian(os.path.splitext(str(path_list[0]))[0])
        # print("JaguarFile, GaussianFile, calc_file_suffix",JaguarFile, GaussianFile, calc_file_suffix)
        if JaguarFile:
            path_list = list(Path(calc_files[0]).glob('*.out'))
        # print(path_list)
        calc_files=[]
        for filename in path_list:
            calc_files.append(os.path.splitext(str(filename))[0])
            #print(os.path.splitext(str(filename))[0], os.path.splitext(str(filename))[1])
            #print(os.path.splitext(str(filename))[0])
            print(os.path.splitext(str(filename))[0], file=logfile)
        # print(calc_files)
        if len(calc_files) == 0:
            print("Error: no files")
            print("Error: no files", file=logfile)
            exit()
        print("Number of files: ",len(calc_files))
        print("Number of files: ",len(calc_files), file=logfile)
# Jaguar or Gaussian?
JaguarFile, GaussianFile, calc_file_suffix = jaguar_or_gaussian(calc_files[0])
if JaguarFile:
    print("Calculated data in Jaguar file")
    print("Calculated data in Jaguar file", file=logfile)
elif GaussianFile:
    print("Calculated data in Gaussian file")
    print("Calculated data in Gaussian file", file=logfile)
else:
    print("Unrecognised calculated data file")
    print("Unrecognised calculated data file", file=logfile)
    exit()

###############################################################################
# Find the energies
calc_energies, dft, basis_set, optimise = get_calc_energy(calc_files, calc_file_suffix, JaguarFile)

# need to sort files by increasing energy, so that the lowest energy structures
# kept as the the unique
# for p in range(0, len(calc_energies)):
#     print(calc_files[p],calc_energies[p])
# print("=============")
pairs = list(zip(calc_energies, calc_files))
sort_pairs = sorted(list(pairs))
# print(sort_pairs)
calc_energies = []
calc_files = []
for p in list(sort_pairs):
    calc_files.append(list(p)[1])
    calc_energies.append(list(p)[0])
    # print("sort check",len(calc_files),list(p)[0],list(p)[1])

# calc_files_unique = [calc_files[0]]
# calc_energies_unique = [calc_energies[0]]
# calc_boltzmann = [1.0]
# calc_boltzmann_unique = [1.0]
calc_files_unique = []
calc_energies_unique = []
calc_boltzmann = []
calc_boltzmann_unique = []
filename = calc_files[0]
calc_spectrum_wavenumbers, calc_spectrum_wnintensity = get_calc_data(filename, minimum_wavenumber, maximum_wavenumber, JaguarFile, calc_file_suffix)
calc_spectrum_wavenumbers_boltz = calc_spectrum_wavenumbers
calc_spectrum_wnintensity_boltz = calc_spectrum_wnintensity
number_of_signals = [len(calc_spectrum_wavenumbers)]
# print("Duplication check ")
# print("Duplication check ", file=logfile)
count_energy_cutoff_rejects = 0
# print(len(calc_files),calc_files)
if len(calc_files) > 1 and boltzmann_analysis:
    for i in range(0, len(calc_files)):
        #print("i ",i, "range: ",range(0, len(calc_files)),calc_files[i].split("/")[-1])
        # print("calc_files_unique: ",len(calc_files_unique),calc_files_unique)
        boltzmann_factor = exp(-(calc_energies[i]-calc_energies[0])/R_in_hartrees/temperature)
        calc_boltzmann.append(boltzmann_factor)
        filename = calc_files[i]
        calc_spectrum_wavenumbers, calc_spectrum_wnintensity = get_calc_data(filename, minimum_wavenumber, maximum_wavenumber, JaguarFile, calc_file_suffix)
        if calc_energies[i]-calc_energies[0] > boltzmann_cutoff_hartree:
            count_energy_cutoff_rejects += 1
        if len(calc_spectrum_wavenumbers) > 0 and calc_energies[i]-calc_energies[0] < boltzmann_cutoff_hartree:
            number_of_signals.append(len(calc_spectrum_wavenumbers))
            unique_structure = True
            for j in range(0,i):
                #print("Checking: ",j,i,calc_energies[j],calc_energies[i])
                if (abs(calc_energies[j]-calc_energies[i])) < max_energy_difference:
                    # energies very similar, so check signals
                    #print("energies similar: ",calc_energies[i],calc_energies[j],calc_files[j].split("/")[-1],calc_files[i].split("/")[-1])
                    # First check the number of signals between minimum_wavenumber and maximum_wavenumber
                    if number_of_signals[i] == number_of_signals[j]:
                        # energies and number of signals the same: quite likely to be identical
                        test_filename = calc_files[j]
                        test_calc_spectrum_wavenumbers, test_calc_spectrum_wnintensity = get_calc_data(test_filename, minimum_wavenumber, maximum_wavenumber, JaguarFile, calc_file_suffix)
                        same_spectra = True
                        #print("energies and number of signals the same: quite likely to be identical ",j,i,number_of_signals[i],len(test_calc_spectrum_wavenumbers),len(calc_spectrum_wavenumbers),test_filename.split("/")[-1])
                        # print(test_calc_spectrum_wavenumbers[0],calc_spectrum_wavenumbers[0],len(calc_spectrum_wavenumbers),len(test_calc_spectrum_wavenumbers))
                        if len(test_calc_spectrum_wavenumbers)*len(calc_spectrum_wavenumbers) == 0:
                            unique_structure = False
                        else:
                            if len(test_calc_spectrum_wavenumbers) != len(calc_spectrum_wavenumbers):
                                same_spectra = False
                            else:
                                for k in range(0,len(calc_spectrum_wavenumbers)):
                                    # print("Wavenumber tests: ",test_filename,k,len(test_calc_spectrum_wavenumbers),len(calc_spectrum_wavenumbers))
                                    # print("Wavenumbers: ",k,test_calc_spectrum_wavenumbers[k],calc_spectrum_wavenumbers[k],same_spectra)
                                    #if abs(test_calc_spectrum_wavenumbers[k]-calc_spectrum_wavenumbers[k]) > 0.5:
                                    if abs(test_calc_spectrum_wavenumbers[k]-calc_spectrum_wavenumbers[k]) > max_vcd_frequency_difference:
                                        same_spectra = False
                                        # print("Not quite a match")
                            if same_spectra:
                                # We have a match! Use the lower energy one
                                #print("We have a match! Use the lower energy one")
                                unique_structure = False
        else:
            number_of_signals.append(1)
            unique_structure = False
            # print("skipping structure ",i,len(calc_energies_unique),len(calc_spectrum_wavenumbers),len(number_of_signals),number_of_signals)
        # print("new file: ",filename,len(calc_spectrum_wavenumbers),len(calc_boltzmann),len(number_of_signals),i,number_of_signals[i],number_of_signals[-1],boltzmann_factor)
        # Is this unique, or should it be ignored?
        # print("checking for unique structure: ",i,calc_files[i],calc_energies[i],unique_structure)
        if unique_structure:
            calc_energies_unique.append(calc_energies[i])
            calc_files_unique.append(calc_files[i])
            calc_boltzmann_unique.append(exp(-(calc_energies[i]-calc_energies[0])/R_in_hartrees/temperature))
            for j in range (0, len(calc_spectrum_wavenumbers)):
                # NB: the combined signals will not be in numerical order, but the
                # order in calc_spectrum_wavenumbers_boltz corresponds to the order
                # in calc_spectrum_wavenumbers_boltz
                calc_spectrum_wavenumbers_boltz.append(calc_spectrum_wavenumbers[j])
                calc_spectrum_wnintensity_boltz.append(calc_spectrum_wnintensity[j]*calc_boltzmann[i])
            # ######
            # print(i,len(calc_files),len(calc_files_unique))
            # print(calc_files)
            # for iii in range(0, len(calc_energies_unique)):
            #     print("     Energy: {:10.6f} hartrees, {:6.3f} kJ/mol, Boltzmann Factor: {:5.3f}".format(calc_energies_unique[iii],(calc_energies_unique[iii]-calc_energies_unique[0])*2625.8,calc_boltzmann_unique[iii]),calc_files_unique[iii])
            # ######

print(count_energy_cutoff_rejects," files rejected by energy cutoff")
print(count_energy_cutoff_rejects," files rejected by energy cutoff", file=logfile)
print(len(calc_energies)-len(calc_energies_unique)-count_energy_cutoff_rejects," duplicate files removed")
print(len(calc_energies)-len(calc_energies_unique)-count_energy_cutoff_rejects," duplicate files removed", file=logfile)

print("Unique Calculated Structures")
print("Unique Calculated Structures", file=logfile)
for i in range(0, len(calc_energies_unique)):
    print("     Energy: {:10.6f} hartrees, {:6.3f} kJ/mol, Boltzmann Factor: {:5.3f}".format(calc_energies_unique[i],(calc_energies_unique[i]-calc_energies_unique[0])*2625.8,calc_boltzmann_unique[i]),calc_files_unique[i], file=logfile)
    print("     Energy: {:10.6f} hartrees, {:6.3f} kJ/mol, Boltzmann Factor: {:5.3f}".format(calc_energies_unique[i],(calc_energies_unique[i]-calc_energies_unique[0])*2625.8,calc_boltzmann_unique[i]),calc_files_unique[i])

one_line_output_decode.extend(["DFT", "Basis set", "optimise"])
one_line_output.extend([dft, basis_set, optimise])
one_line_output_decode.extend(["Minimum wavenumber", "Maximum wavenumber", "Temperature (K)"])
one_line_output.extend([minimum_wavenumber, maximum_wavenumber, temperature])
one_line_output_decode.extend(["Line broadening", "Number of conformations", "Number of unique conformations"])
one_line_output.extend([hwhm_lor_broad, len(calc_energies), len(calc_energies_unique)])
one_line_output_decode.append("Boltzmann Analysis")
one_line_output.append(boltzmann_analysis)

if boltzmann_analysis:
    print("Using all ",len(calc_files_unique),"unique conformations within energy cut-off")
    print("Using all ",len(calc_files_unique),"unique conformations within energy cut-off", file=logfile)
else:
    # calc_energies has been sorted, so calc_energies[0] has the largest Boltzmann Factor
    print("Using only lowest energy conformation in analysis: ",calc_files_unique[0]," Energy: ",calc_energies_unique[0])
    print("Using only lowest energy conformation in analysis: ",calc_files_unique[0]," Energy: ",calc_energies_unique[0], file=logfile)

###############################################################################
# Check on all analyses
# max_confidence list has values for:
# (i) Alpha only
# (ii) Beta only
# (iii) Alpha and beta - this one is probably not useful
# (iv) Corrected (difference between alpha and beta)
#
max_confidence = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
preferred_scaling_factor = [defined_scaling_factor, defined_scaling_factor, defined_scaling_factor, defined_scaling_factor, defined_scaling_factor, defined_scaling_factor]
best_match_score = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
match_score = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
confidence = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
bsf_match_score = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
bsf_confidence = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
average_match_score = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
average_confidence = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
max_average_confidence = [-99999, -99999, -99999, -99999, -99999, -99999]
min_average_confidence = [99999, 99999, 99999, 99999, 99999, 99999]
max_average_sf = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
min_average_sf = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
count_average_match_score = 0.0

###############################################################################
# Scaling factor optimisation
if print_scaling_factor_csv:
    confidence_csvfile = open(sys.argv[1].split(".")[0]+"_confidence.csv", 'w')
    print("Scaling Factor, Alpha sum, Alpha scale, Beta sum, Beta scale, Corr sum, Corr scale, Confidence Alpha, Confidence Beta, Confidence Both, Confidence Corr", file=confidence_csvfile)
for scaling_factor in [0.950, 0.955, 0.960, 0.965, 0.966, 0.967, 0.968, 0.969, 0.970, 0.971, 0.972, 0.973, 0.974, 0.975, 0.976, 0.977, 0.978, 0.979, 0.980, 0.981, 0.982, 0.983, 0.984, 0.985, 0.986, 0.987, 0.988, 0.989, 0.990, 0.991, 0.992, 0.993, 0.994, 0.995, 0.996, 0.997, 0.998, 0.999, 1.000]:
    calc_signals = lorentzian(expt_wavenums,calc_spectrum_wavenumbers_boltz,calc_spectrum_wnintensity_boltz,hwhm_lor_broad,scaling_factor)
    match_score[0] = match_score_calc(calc_signals,alfa_expt_signals)
    match_score[1] = match_score_calc(calc_signals,beta_expt_signals)
    match_score[2] = [match_score[0][0] - match_score[1][0], match_score[0][1] - match_score[1][1]]
    match_score[3] = match_score_calc(calc_signals,corr_alfa_expt_signals)
    match_score[4] = match_score_calc(calc_signals,alfa_expt_signals_blank_correction)
    match_score[5] = match_score_calc(calc_signals,beta_expt_signals_blank_correction)
    confidence[0] = 100.0 * match_score[0][0]
    confidence[1] = 100.0 * match_score[1][0]
    # There should be a better way to do Alpha/Beta;
    # If they are both high, the combination should be higher
    # How about sqrt(1-(1-a)(1-b)) = sqrt(a+b-ab) for 0<a,b<1 ?
    confidence[2] = 50.0 * abs(match_score[0][0] - match_score[1][0])
    confidence[3] = 100.0 * match_score[3][0]
    confidence[4] = 100.0 * match_score[4][0]
    confidence[5] = 100.0 * match_score[5][0]
    if abs(scaling_factor - defined_scaling_factor) < extreme_sf_warning:
        count_average_match_score += 1.0
        for i in range(0, len(match_score)):
            average_match_score[i] += match_score[i][1]
            average_confidence[i] += confidence[i]
            if max_average_confidence[i] < confidence[i]:
                max_average_confidence[i] = confidence[i]
                max_average_sf[i] = scaling_factor
            if min_average_confidence[i] > confidence[i]:
                min_average_confidence[i] = confidence[i]
                min_average_sf[i] = scaling_factor
    for i in range(0, len(match_score)):
        if abs(confidence[i]) > abs(max_confidence[i]):
            max_confidence[i] = confidence[i]
            preferred_scaling_factor[i] = scaling_factor
            best_match_score[i] = [match_score[i][0], match_score[i][1]]
    if print_scaling_factor_csv:
        print(scaling_factor, ",", match_score[0][1], ",", match_score[0][0],
            ",", match_score[1][1], ",", match_score[1][0], ",",
            match_score[3][1], ",", match_score[3][0], ",", confidence[0],
            ",", confidence[1], ",", confidence[2], ",", confidence[3], file=confidence_csvfile)
for i in range(0, len(match_score)):
    average_match_score[i] = average_match_score[i]/count_average_match_score
    average_confidence[i] = average_confidence[i]/count_average_match_score
print()
print("", file=logfile)
if print_scaling_factor_csv:
    confidence_csvfile.close()

# Now calculate for defined_scaling_factor
calc_signals = lorentzian(expt_wavenums,calc_spectrum_wavenumbers_boltz,calc_spectrum_wnintensity_boltz,hwhm_lor_broad,defined_scaling_factor)
bsf_match_score[0] = match_score_calc(calc_signals,alfa_expt_signals)
bsf_match_score[1] = match_score_calc(calc_signals,beta_expt_signals)
bsf_match_score[2] = [bsf_match_score[0][0] - bsf_match_score[1][0], bsf_match_score[0][1] - bsf_match_score[1][1]]
bsf_match_score[3] = match_score_calc(calc_signals,corr_alfa_expt_signals)
bsf_match_score[4] = match_score_calc(calc_signals,alfa_expt_signals_blank_correction)
bsf_match_score[5] = match_score_calc(calc_signals,beta_expt_signals_blank_correction)
bsf_confidence[0] = 100.0 * bsf_match_score[0][0]
bsf_confidence[1] = 100.0 * bsf_match_score[1][0]
bsf_confidence[2] = 50.0 * abs(bsf_match_score[0][0] + bsf_match_score[1][0])
bsf_confidence[3] = 100.0 * bsf_match_score[3][0]
bsf_confidence[4] = 100.0 * bsf_match_score[4][0]
bsf_confidence[5] = 100.0 * bsf_match_score[5][0]
print()
print("", file=logfile)

if print_spectra_csv:
    csvfile = open(sys.argv[1].split(".")[0]+".csv", 'w')
    print("#, Experimental Wavenumbers, Blank File, Alfa experiment, Beta experiment, Corrected Alfa, Calculated Signals", file=csvfile)
    for i in range(0, len(alfa_expt_signals)):
        print(i,",",expt_wavenums[i],",",blank_file_intensity[i],",",alfa_expt_signals[i],",",beta_expt_signals[i],",",
            corr_alfa_expt_signals[i],",",calc_signals[i], file=csvfile)
    csvfile.close()


print("Defined scaling factor:",defined_scaling_factor)
print("Defined scaling factor:",defined_scaling_factor, file=logfile)
output_labels = ["File (A):", "File (B):", "Alpha/Beta:", "Combined:", "File (A) blank:", "File (B) blank:"]
sf_label = "Defined SF"
one_line_output_decode.append("Defined Scale Factor")
one_line_output.append(defined_scaling_factor)
#for i in range(0, len(bsf_match_score)):
if single_enantiomer:
    printout_confidence("Single enantiomer result:", sf_label, defined_scaling_factor, bsf_match_score[0][0], bsf_confidence[0])
    one_line_output_decode.extend([output_labels[0],output_labels[1],output_labels[2],output_labels[3]])
    one_line_output.extend([bsf_confidence[0],"","",""])
    one_line_output_decode.extend([output_labels[4],output_labels[5]])
    if blank_file_marker:
        one_line_output.extend([bsf_confidence[4],""])
        printout_confidence("Single enantiomer blank: ", sf_label, defined_scaling_factor, bsf_match_score[4][0], bsf_confidence[4])
    else:
        one_line_output.extend(["",""])
else:
    printout_confidence(output_labels[0], sf_label, defined_scaling_factor, bsf_match_score[0][0], bsf_confidence[0])
    printout_confidence(output_labels[1], sf_label, defined_scaling_factor, bsf_match_score[1][0], bsf_confidence[1])
    if blank_file_marker:
        printout_confidence(output_labels[4], sf_label, defined_scaling_factor, bsf_match_score[4][0], bsf_confidence[4])
        printout_confidence(output_labels[5], sf_label, defined_scaling_factor, bsf_match_score[5][0], bsf_confidence[5])
    printout_confidence(output_labels[3], sf_label, defined_scaling_factor, bsf_match_score[3][0], bsf_confidence[3])
    for i in range(0, len(bsf_match_score)):
        one_line_output_decode.append(output_labels[i])
        one_line_output.append(bsf_confidence[i])
print("\n")
print("\n", file=logfile)

sf_label = "Opt. SF"
if single_enantiomer:
    # printout_confidence("Single enantiomer result:", preferred_scaling_factor[0], match_score[0][0], max_confidence[0])
    printout_confidence("Single enantiomer result:", sf_label, preferred_scaling_factor[0], best_match_score[0][0], max_confidence[0])
    one_line_output_decode.extend(["Optimised Scale Factor: "+output_labels[0],"Confidence: "+output_labels[0],"Optimised Scale Factor: "+output_labels[1],"Confidence: "+output_labels[1],"Optimised Scale Factor: "+output_labels[2],"Confidence: "+output_labels[2],"Optimised Scale Factor: "+output_labels[3],"Confidence: "+output_labels[3]])
    one_line_output_decode.extend(["Optimised Scale Factor: "+output_labels[4],"Confidence: "+output_labels[4],"Optimised Scale Factor: "+output_labels[5],"Confidence: "+output_labels[5]])
    one_line_output.extend([preferred_scaling_factor[0],max_confidence[0],"","","","","",""])
    if blank_file_marker:
        one_line_output.extend([max_confidence[4],"","",""])
        printout_confidence("Single enantiomer blank: ", sf_label, preferred_scaling_factor[4], best_match_score[4][0], max_confidence[4])
    else:
        one_line_output.extend(["","","",""])
else:
    #printout_confidence(output_labels[i], sf_label, preferred_scaling_factor[i], best_match_score[i][0], max_confidence[i])
    printout_confidence(output_labels[0], sf_label, preferred_scaling_factor[0], best_match_score[0][0], max_confidence[0])
    if abs(preferred_scaling_factor[0]-defined_scaling_factor) > extreme_sf_warning:
        print("   *** Warning *** optimised scale factor rather extreme")
        print("   *** Warning *** optimised scale factor rather extreme", file=logfile)
    printout_confidence(output_labels[1], sf_label, preferred_scaling_factor[1], best_match_score[1][0], max_confidence[1])
    if abs(preferred_scaling_factor[1]-defined_scaling_factor) > extreme_sf_warning:
        print("   *** Warning *** optimised scale factor rather extreme")
        print("   *** Warning *** optimised scale factor rather extreme", file=logfile)
    if blank_file_marker:
        printout_confidence(output_labels[4], sf_label, preferred_scaling_factor[4], best_match_score[4][0], max_confidence[4])
        if abs(preferred_scaling_factor[4]-defined_scaling_factor) > extreme_sf_warning:
            print("   *** Warning *** optimised scale factor rather extreme")
            print("   *** Warning *** optimised scale factor rather extreme", file=logfile)
        printout_confidence(output_labels[5], sf_label, preferred_scaling_factor[5], best_match_score[5][0], max_confidence[5])
        if abs(preferred_scaling_factor[5]-defined_scaling_factor) > extreme_sf_warning:
            print("   *** Warning *** optimised scale factor rather extreme")
            print("   *** Warning *** optimised scale factor rather extreme", file=logfile)
    printout_confidence(output_labels[3], sf_label, preferred_scaling_factor[3], best_match_score[3][0], max_confidence[3])
    if abs(preferred_scaling_factor[3]-defined_scaling_factor) > extreme_sf_warning:
        print("   *** Warning *** optimised scale factor rather extreme")
        print("   *** Warning *** optimised scale factor rather extreme", file=logfile)
    for i in range(0, len(preferred_scaling_factor)):
        one_line_output_decode.append("Optimised Scale Factor: "+output_labels[i])
        one_line_output.append(preferred_scaling_factor[i])
        one_line_output_decode.append("Confidence: "+output_labels[i])
        one_line_output.append(max_confidence[i])
print()
print("", file=logfile)

# print("Average result over scaling factor range:",defined_scaling_factor-extreme_sf_warning,defined_scaling_factor+extreme_sf_warning)
print("Average result over scaling factor range:",defined_scaling_factor-extreme_sf_warning,defined_scaling_factor+extreme_sf_warning, file=logfile)
output_labels = ["av. File (A):", "av. File (B):", "av. Alpha/Beta:", "av. Combined:", "av. File (A) blank:", "av. File (B) blank:"]
sf_label = "Average result over SF range"
one_line_output_decode.append("Average Scale Factor")
one_line_output.append(defined_scaling_factor)
#for i in range(0, len(bsf_match_score)):
if single_enantiomer:
    # printout_confidence("Single enantiomer result:", sf_label, defined_scaling_factor, average_confidence[0], average_confidence[0])
    # printout_confidence("Single enantiomer max:   ", " ", max_average_sf[0], max_average_confidence[0], max_average_confidence[0])
    # printout_confidence("Single enantiomer min:   ", " ", min_average_sf[0], min_average_confidence[0], min_average_confidence[0])
    one_line_output_decode.extend([output_labels[0],output_labels[1],output_labels[2],output_labels[3]])
    one_line_output.extend([average_confidence[0],"","",""])
    one_line_output_decode.extend([output_labels[4],output_labels[5]])
    if blank_file_marker:
        one_line_output.extend([average_confidence[4],""])
        # printout_confidence("Single enantiomer blank: ", sf_label, defined_scaling_factor, average_match_score[4], average_confidence[4])
    else:
        one_line_output.extend(["",""])
else:
    # printout_confidence(output_labels[0], sf_label, defined_scaling_factor, average_confidence[0], average_confidence[0])
    # printout_confidence(output_labels[0]+" max", " ", max_average_sf[0], max_average_confidence[0], max_average_confidence[0])
    # printout_confidence(output_labels[0]+" min", " ", min_average_sf[0], min_average_confidence[0], min_average_confidence[0])
    # printout_confidence(output_labels[1], sf_label, defined_scaling_factor, average_confidence[1], average_confidence[1])
    # printout_confidence(output_labels[1]+" max", " ", max_average_sf[1], max_average_confidence[1], max_average_confidence[1])
    # printout_confidence(output_labels[1]+" min", " ", min_average_sf[1], min_average_confidence[1], min_average_confidence[1])
    # if blank_file_marker:
    #     printout_confidence(output_labels[4], sf_label, defined_scaling_factor, average_confidence[4], average_confidence[4])
    #     printout_confidence(output_labels[4]+" max", " ", max_average_sf[4], max_average_confidence[4], max_average_confidence[4])
    #     printout_confidence(output_labels[4]+" min", " ", min_average_sf[4], min_average_confidence[4], min_average_confidence[4])
    #     printout_confidence(output_labels[5], sf_label, defined_scaling_factor, average_confidence[5], average_confidence[5])
    #     printout_confidence(output_labels[5]+" max", " ", max_average_sf[5], max_average_confidence[5], max_average_confidence[5])
    #     printout_confidence(output_labels[5]+" min", " ", min_average_sf[5], min_average_confidence[5], min_average_confidence[5])
    # printout_confidence(output_labels[3], sf_label, defined_scaling_factor, average_confidence[3], average_confidence[3])
    # printout_confidence(output_labels[3]+" max", " ", max_average_sf[3], max_average_confidence[3], max_average_confidence[3])
    # printout_confidence(output_labels[3]+" min", " ", min_average_sf[3], min_average_confidence[3], min_average_confidence[3])
    for i in range(0, len(bsf_match_score)):
        one_line_output_decode.append(output_labels[i])
        one_line_output.append(average_confidence[i])
# print("\n")
print("\n", file=logfile)




if single_enantiomer:
    # print("Single enantiomer: ",alfa_expt_file)
    print("Experimental data for single enantiomer:\n(A) ",alfa_expt_file)
    print("Experimental data for single enantiomer:\n(A) ",alfa_expt_file, file=logfile)
else:
    print("Experimental data for both enantiomers:\n(A) ",alfa_expt_file,"\n(B) ",beta_expt_file)
    print("Experimental data for both enantiomers:\n(A) ",alfa_expt_file,"\n(B) ",beta_expt_file, file=logfile)
if blank_file_marker:
    print("Blank file:\n",blank_file)
    print("Blank file:\n",blank_file, file=logfile)
print()
print("", file=logfile)

###############################################################################
# Summary
total_confidence, goodness_description = summarise_conclusion()

###############################################################################
# Print graphs of the results
if print_graph:
    print_graph_files(total_confidence, goodness_description)

if print_summary_csv:
    print_tlo_summary(one_line_output_decode, one_line_output)
logfile.close()
