import gpype as gp
import time
"""
Potentially useful methods
gp.MovingAverage(window_size): power bands. power can fluctuate a lot, moving average helps smooth
gp.Decimator(decimation_factor): reduce sampling rate (to less than 250Hz), saves CPU
gp.Hold(): keeps last valid sample in buffer to avoid 'None'

gp.Router(input_channels): select specific channels
gp.Router.ALL to pass all channels
gp.Framer(frame_size): groups samples into chunks ('frames'), saves CPU
gp.TimeSeriesScope(): real time graph, can add markers to it
gp.CsvWriter(file_name): if we want to write data
gp.LSLSender(): to send to another program using lab streaming layer

gp.FFT(): if we want the raw frequency spectrum (less black box)
gp.Equation(expression): e.g. in1 / (in1 + in2) to calculate alpha/beta ration directly in pipeline
"""
def calibrate(alpha_node, duration=10):
    samples = []
    start_time = time.time()

    while time.time() - start_time < duration:
        val = alpha_node.get_value()
        if val > 0: #don't want 0s in startup
            samples.append(val)
        time.sleep(0.1) #CPU catchup

    baseline = sum(samples) / len(samples) if samples else 1.0
    return baseline

def main():
    #Initialize
    app = gp.MainApp()
    pipeline = gp.Pipeline()

    #Connect to Unicorn Headset
    source = gp.UnicornSource(device_id=0)
    pipeline.add_node(source)

    #Pre-process
    bandpass = gp.BandpassFilter(f_low=1.0, f_high=50.0, order =4)
    notch = gp.NotchFilter(f_center=50.0, bandwidth=2.0)

    pipeline.add_node(bandpass, input_node = source)
    pipeline.add_node(notch, input_node=bandpass)

    frontal_node = gp.Router(input_channels=[0]) #Fz
    occipital_node = gp.Router(input_channels=[6]) #Oz

    pipeline.add_node(frontal_node, input_node=notch)
    pipeline.add_node(occipital_node, input_node=notch)

    #Power band extraction
    theta_power = gp.ThetaPower(smoothing=0.5) #4-8Hz
    alpha_power = gp.AlphaPower(smoothing=0.5) #8-13Hz
    beta_power = gp.BetaPower(smoothing=0.5) #13-30Hz

    pipeline.add_node(theta_power, input_node=frontal_node)
    pipeline.add_node(alpha_power, input_node=occipital_node)
    pipeline.add_node(beta_power, input_node=source)

    #Moving averages
    theta_smooth = gp.MovingAverage(window_size=250) #1 sec avg
    pipeline.add_node(theta_smooth, input_node=theta_power)

    alpha_smooth = gp.MovingAverage(window_size=250) #1 sec avg
    pipeline.add_node(alpha_smooth, input_node=alpha_power)

    beta_smooth = gp.MovingAverage(window_size=250) #1 sec avg
    pipeline.add_node(beta_smooth, input_node=beta_power)
    #Start stream
    pipeline.start()

    alpha_baseline = calibrate(alpha_smooth, duration = 10)
    
    try:
        print(f"{'Theta':^10} | {'Alpha':^10} | {'Beta':^10}")
        print("-" * 36)

        while True:
            #pull latest values from nodes
            t_val = theta_power.get_value()
            a_val = alpha_power.get_value()
            b_val = beta_power.get_value()

            a_rel = a_val / alpha_baseline
            

            print(f"{t_val*100:10.2f} | {a_val*100:10.2f} | {b_val*100:10.2f}", end='\r')

            time.sleep(0.1)

    except KeyboardInterrupt:
        pipeline.stop()


if __name__ == "__main__":
    main()