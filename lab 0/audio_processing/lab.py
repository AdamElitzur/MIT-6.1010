"""
6.101 Lab:
Audio Processing
"""

import wave
import struct

# No additional imports allowed!


def backwards(sound):
    """
    Returns a new sound containing the samples of the original in reverse
    order, without modifying the input sound.

    Args:
        sound: a dictionary representing the original mono sound

    Returns:
        A new mono sound dictionary with the samples in reversed order
    """
    new_sound = sound.copy()
    new_sound["samples"] = sound["samples"][::-1] # revesing using slices
    return new_sound


def mix(sound1, sound2, p):
    """
    Takes 2 sounds, mixes them with a mixing parameter, 
    takes p times the first param and 1-p times the second parameter

    parameters:
    * sound1, sound2: two different sounds as a dictionary
    * p: a number from 0-1

    returns:
    * a new sound mixed with p
    """
    # mix 2 good sounds
    if (
        "rate" not in sound1 # fixed this part "rate" in sound1 == ....
        or "rate" not in sound2
        or sound1["rate"] != sound2["rate"]
    ):

        print("no")
        return None

    r = sound1["rate"]  # get rate
    sound1 = sound1["samples"]
    sound2 = sound2["samples"]
    sound_length = max(len(sound1), len(sound2))
    # removed next few lines for this max comparison
    # if len(sound1) < len(sound2):
    #     sound_length = len(sound2)
    # elif len(sound2) < len(sound1):
    #     sound_length = len(sound1)
    # elif len(sound1) == len(sound2):
    #     sound_length = len(sound2)
    # else:
    #     print("whoops")
    #     return

    new_samples = []
    x = 0
    while x <= sound_length:
        # can I do anything to remove these if statements (like the one above)?
        if x < len(sound1) and x < len(sound2):
            new_samples.append(p * sound1[x] + sound2[x] * (1 - p))
        # elif x < len(sound1) and x >= len(sound2):
        elif len(sound1) > x >= len(sound2):
            new_samples.append(p * sound1[x])
        # elif x >= len(sound1) and x < len(sound2):
        elif len(sound1) <= x < len(sound2): # fixed spacing issues
            new_samples.append(sound2[x] * (1 - p))
        # elif x >= len(sound1) and x >= len(sound2):
        #     new_samples.append(0)
        x += 1
        if x == sound_length:  # end
            break
    final_sound = {"rate": r, "samples": new_samples}
    return final_sound  # return new sound


def convolve(sound, kernel):
    """
    Compute a new sound by convolving the given input sound with the given
    kernel.  Does not modify input sound.

    Args:
        sound: a dictionary representing the original mono sound
        kernel: list of numbers, the signal with which the sound should be
                convolved

    Returns:
        A new mono sound dictionary resulting from the convolution.
        
    """
    samples = sound["samples"]
    final = [0] * (len(samples)+len(kernel)-1) # initialize final list of zeroes
    temp_list = []
    for index, value in enumerate(kernel): # looping over the kernel
        if value != 0:
            # make a temp_list with index # of 0s, then multiply that kernel index
            # by every sample (scaling factor)
            temp_list = ([0] * index) + [j*kernel[index] for j in samples]
            for k, value in enumerate(temp_list):
                # adding in all the different additions
                final[k] += value
    # previous code with range, instead of enumerate
    # for i in range(len(kernel)):
    #     if kernel[i] != 0:
    #         temp_list = ([0] * i) + [j*kernel[i] for j in samples]
    #         for k in range(len(temp_list)): # adding in all the different additions
    #             final[k] += temp_list[k]
    return {"rate": sound["rate"], "samples": final}


def echo(sound, num_echoes, delay, scale):
    """
    Compute a new sound consisting of several scaled-down and delayed versions
    of the input sound. Does not modify input sound.

    Args:
        sound: a dictionary representing the original mono sound
        num_echoes: int, the number of additional copies of the sound to add
        delay: float, the amount of seconds each echo should be delayed
        scale: float, the amount by which each echo's samples should be scaled

    Returns:
        A new mono sound dictionary resulting from applying the echo effect.
    """
    sample_delay = round(delay * sound["rate"])

    # creates a kernel that is spaced evenly
    # ex: [1,0,0,.2,0,0,.4] based on sample_delay of 3, scale of .2
    kernel = [1] + [0] * (sample_delay-1) + [scale]
    for i in range(1, num_echoes):
        kernel += [0] * (sample_delay-1) + [scale**(i+1)]

    # uses this new kernel to convolve and create an echo
    final = convolve(sound, kernel)
    return final

def pan(sound):
    """
    pans audio from left to right by changing volume.

    input: sound with right and left channel

    output: new sound with right and left channel, panning
    """
    left = []
    right = []
    length = len(sound["left"])

    # first entries for right and left bc they don't follow convention
    right.append(0)
    left.append(sound["left"][0])

    # looping over the rest appending them with the convention
    for i in range(1, length):
        right.append(sound["right"][i] * (i/(length-1)))
        left.append(sound["left"][i] * (1-i/(length-1)))

    return {"rate": sound["rate"], "left": left, "right": right}


def remove_vocals(sound):
    """
    subtract left from right
    """
    final_sample = []
    length = len(sound["left"])
    for i in range(length):
        # subtract left from right in loop
        final_sample.append(sound["left"][i]-sound["right"][i])
    return {"rate": sound["rate"], "samples": final_sample}


def bass_boost_kernel(n_val, scale=0):
    """
    Construct a kernel that acts as a bass-boost filter.

    We start by making a low-pass filter, whose frequency response is given by
    (1/2 + 1/2cos(Omega)) ^ n_val

    Then we scale that piece up and add a copy of the original signal back in.
    """
    # make this a fake "sound" so that we can use the convolve function
    base = {"rate": 0, "samples": [0.25, 0.5, 0.25]}
    kernel = {"rate": 0, "samples": [0.25, 0.5, 0.25]}
    for i in range(n_val):
        kernel = convolve(kernel, base["samples"])
    kernel = kernel["samples"]

    # at this point, the kernel will be acting as a low-pass filter, so we
    # scale up the values by the given scale, and add in a value in the middle
    # to get a (delayed) copy of the original
    kernel = [i * scale for i in kernel]
    kernel[len(kernel) // 2] += 1

    return kernel


# below are helper functions for converting back-and-forth between WAV files
# and our internal dictionary representation for sounds


def load_wav(filename, stereo=False):
    """
    Given the filename of a WAV file, load the data from that file and return a
    Python dictionary representing that sound
    """
    file = wave.open(filename, "r")
    chan, bd, sr, count, _, _ = file.getparams()

    assert bd == 2, "only 16-bit WAV files are supported"

    out = {"rate": sr}

    if stereo:
        left = []
        right = []
        for i in range(count):
            frame = file.readframes(1)
            if chan == 2:
                left.append(struct.unpack("<h", frame[:2])[0])
                right.append(struct.unpack("<h", frame[2:])[0])
            else:
                datum = struct.unpack("<h", frame)[0]
                left.append(datum)
                right.append(datum)

        out["left"] = [i / (2**15) for i in left]
        out["right"] = [i / (2**15) for i in right]
    else:
        samples = []
        for i in range(count):
            frame = file.readframes(1)
            if chan == 2:
                left = struct.unpack("<h", frame[:2])[0]
                right = struct.unpack("<h", frame[2:])[0]
                samples.append((left + right) / 2)
            else:
                datum = struct.unpack("<h", frame)[0]
                samples.append(datum)

        out["samples"] = [i / (2**15) for i in samples]

    return out


def write_wav(sound, filename):
    """
    Given a dictionary representing a sound, and a filename, convert the given
    sound into WAV format and save it as a file with the given filename (which
    can then be opened by most audio players)
    """
    outfile = wave.open(filename, "w")

    if "samples" in sound:
        # mono file
        outfile.setparams((1, 2, sound["rate"], 0, "NONE", "not compressed"))
        out = [int(max(-1, min(1, v)) * (2**15 - 1)) for v in sound["samples"]]
    else:
        # stereo
        outfile.setparams((2, 2, sound["rate"], 0, "NONE", "not compressed"))
        out = []
        for left, right in zip(sound["left"], sound["right"]):
            left = int(max(-1, min(1, left)) * (2**15 - 1))
            right = int(max(-1, min(1, right)) * (2**15 - 1))
            out.append(left)
            out.append(right)

    outfile.writeframes(b"".join(struct.pack("<h", frame) for frame in out))
    outfile.close()


if __name__ == "__main__":
    # code in this block will only be run when you explicitly run your script,
    # and not when the tests are being run.  this is a good place to put your
    # code for generating and saving sounds, or any other code you write for
    # testing, etc.

    # here is an example of loading a file (note that this is specified as
    # sounds/hello.wav, rather than just as hello.wav, to account for the
    # sound files being in a different directory than this file)
    mystery = load_wav("sounds/mystery.wav")
    write_wav(backwards(mystery), "mystery_backwards.wav")

    synth = load_wav("sounds/synth.wav")
    water = load_wav("sounds/water.wav")
    write_wav(mix(synth, water, 0.2), "syth_water_mix.wav")

    test_kernel = bass_boost_kernel(1000, 1.5)
    ice_and_chilli = load_wav("sounds/ice_and_chilli.wav")
    write_wav(convolve(ice_and_chilli, test_kernel), "ice_and_chilli_bass_boosted.wav")

    chord = load_wav("sounds/chord.wav")
    write_wav(echo(chord, 5, 0.3, 0.6), "chord_echo.wav")

    car = load_wav("sounds/car.wav", stereo=True)
    write_wav(pan(car), "car_panned.wav")

    mountain = load_wav("sounds/lookout_mountain.wav", stereo=True)
    write_wav(remove_vocals(mountain), "mountain_no_vocal.wav")
