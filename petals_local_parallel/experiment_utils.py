import pandas as pd

def update_results(res_df: pd.DataFrame, rank: int,
                   do_backward: int,
                   num_iter: int,
                   batch_size: int,
                   seq_length: int,
                   hid_size: int,
                   model_type: str,
                   config: str,
                   device_name: str,
                   working_time) -> pd.DataFrame():

    res_df[f"{rank}_do_backward"] = do_backward
    res_df[f"{rank}_num_iter"]    = num_iter
    res_df[f"{rank}_batch_size"]  = batch_size
    res_df[f"{rank}_seq_length"]  = seq_length
    res_df[f"{rank}_config"]      = config  # bloom config
    res_df[f"{rank}_hid_size"]    = hid_size
    res_df[f"{rank}_model_type"]  = model_type  # only mlp, only attention, both
    res_df[f"{rank}_device_name"] = device_name  # which gpu we are using

    # Store the computation time.
    res_df[f"{rank}_working_time"] = working_time

    return res_df