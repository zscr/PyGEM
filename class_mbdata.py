"""class of mass balance data and functions associated with manipulating the dataset to be in the proper format"""

# External libraries
import pandas as pd
import numpy as np
import calendar
import datetime
# Local libraries
import pygem_input as input
import pygemfxns_modelsetup as modelsetup


class MBData():
    """
    Mass balance data properties and functions used to automatically retrieve data for calibration.
    
    Attributes
    ----------
    name : str
        name of mass balance dataset.
    rgi_regionO1 : int
        number of RGI order 1 region.
    ds_fp : str
        file path 
    """
    def __init__(self, 
                 name='wgms_d',
                 rgi_regionO1=input.rgi_regionsO1[0]
                 ):
        """
        Add variable name and specific properties associated with each variable.
        """
        
        # Source of climate data
        self.name = name
        # Set parameters for ERA-Interim and CMIP5 netcdf files
        if self.name == 'shean': 
            self.rgi_regionO1 = rgi_regionO1
            self.ds_fp = input.shean_fp
            self.ds_fn = input.shean_fn
            self.rgi_glacno_cn = input.shean_rgi_glacno_cn
            self.mb_mwea_cn = input.shean_mb_cn
            self.mb_mwea_err_cn = input.shean_mb_err_cn
            self.t1_cn = input.shean_time1_cn
            self.t2_cn = input.shean_time2_cn
            self.area_cn = input.shean_area_cn
            
        elif self.name == 'mcnabb':
            self.rgi_regionO1 = rgi_regionO1
            self.ds_fp = input.mcnabb_fp
            self.ds_fn = input.mcnabb_fn
            self.rgiid = input.mcnabb_rgiid_cn
            self.mb_mwea_cn = input.mcnabb_mb_cn
            self.mb_mwea_err_cn = input.mcnabb_mb_err_cn
            self.t1_cn = input.mcnabb_time1_cn
            self.t2_cn = input.mcnabb_time2_cn
            self.area_cn = input.mcnabb_area_cn
        
        elif self.name == 'larsen':
            self.rgi_regionO1 = rgi_regionO1
            self.ds_fp = input.larsen_fp
            self.ds_fn = input.larsen_fn
            self.rgiid = input.larsen_rgiid_cn
            self.mb_mwea_cn = input.larsen_mb_cn
            self.mb_mwea_err_cn = input.larsen_mb_err_cn
            self.t1_cn = input.larsen_time1_cn
            self.t2_cn = input.larsen_time2_cn
            self.area_cn = input.larsen_area_cn
        
        elif self.name == 'brun':
            self.data_fp = input.brun_fp
            
        elif self.name == 'mauer':
            self.rgi_regionO1 = rgi_regionO1
            self.ds_fp = input.mauer_fp
            self.ds_fn = input.mauer_fn
            self.rgi_glacno_cn = input.mauer_rgi_glacno_cn
            self.mb_mwea_cn = input.mauer_mb_cn
            self.mb_mwea_err_cn = input.mauer_mb_err_cn
            self.t1_cn = input.mauer_time1_cn
            self.t2_cn = input.mauer_time2_cn
            
        elif self.name == 'wgms_ee':
            self.rgi_regionO1 = rgi_regionO1
            self.ds_fp = input.wgms_fp
            self.ds_fn = input.wgms_ee_fn_preprocessed
            self.rgi_glacno_cn = input.wgms_rgi_glacno_cn
            self.mb_mwe_cn = input.wgms_ee_mb_cn
            self.mb_mwe_err_cn = input.wgms_ee_mb_err_cn
            self.t1_cn = input.wgms_ee_t1_cn
            self.period_cn = input.wgms_ee_period_cn
            self.z1_cn = input.wgms_ee_z1_cn
            self.z2_cn = input.wgms_ee_z2_cn
            self.obs_type_cn = input.wgms_obs_type_cn
            
        elif self.name == 'wgms_d':
            self.rgi_regionO1 = rgi_regionO1
            self.ds_fp = input.wgms_fp
            self.ds_fn = input.wgms_d_fn_preprocessed
            self.rgi_glacno_cn = input.wgms_rgi_glacno_cn
            self.thickness_chg_cn = input.wgms_d_thickness_chg_cn
            self.thickness_chg_err_cn = input.wgms_d_thickness_chg_err_cn
            self.volume_chg_cn = input.wgms_d_volume_chg_cn
            self.volume_chg_err_cn = input.wgms_d_volume_chg_err_cn
            self.z1_cn = input.wgms_d_z1_cn
            self.z2_cn = input.wgms_d_z2_cn
            self.obs_type_cn = input.wgms_obs_type_cn
            
        elif self.name == 'cogley':
            self.rgi_regionO1 = rgi_regionO1
            self.ds_fp = input.cogley_fp
            self.ds_fn = input.cogley_fn_preprocessed
            self.rgi_glacno_cn = input.cogley_rgi_glacno_cn
            self.mass_chg_cn = input.cogley_mass_chg_cn
            self.mass_chg_err_cn = input.cogley_mass_chg_err_cn
            self.z1_cn = input.cogley_z1_cn
            self.z2_cn = input.cogley_z2_cn
            self.obs_type_cn = input.cogley_obs_type_cn
            
        elif self.name == 'group':
            self.rgi_regionO1 = rgi_regionO1
            self.ds_fp = input.mb_group_fp
            self.ds_fn = input.mb_group_data_fn
            self.ds_dict_fn = input.mb_group_dict_fn
            self.rgi_regionO1_cn = 'rgi_regionO1'
            self.t1_cn = input.mb_group_t1_cn
            self.t2_cn = input.mb_group_t2_cn
            
            
            
    def retrieve_mb(self, main_glac_rgi, main_glac_hyps, dates_table):
        """
        Retrieve the mass balance for various datasets to be used in the calibration.
        
        
        Parameters
        ----------
        main_glac_rgi : pandas dataframe
            dataframe containing relevant rgi glacier information
        main_glac_hyps : pandas dataframe
            dataframe containing glacier hypsometry
        dates_table : pandas dataframe
            dataframe containing dates of model run
        
        Returns
        -------
        ds_output : pandas dataframe
            dataframe of mass balance observations and other relevant information for calibration 
        """       
        # Column names of output
        ds_output_cols = ['RGIId', 'glacno', 'group_name', 'obs_type', 'mb_mwe', 'mb_mwe_err', 'sla_m',  'z1_idx', 
                          'z2_idx', 'z1', 'z2', 't1_idx', 't2_idx', 't1', 't2', 'area_km2', 'WGMS_ID']
        # Reset rgi index so it is consistent with hyps and other data
        main_glac_rgi.reset_index(drop=True, inplace=True)
        # Dictionary linking glacier number (glacno) to index
        glacnodict = dict(zip(main_glac_rgi['glacno'], main_glac_rgi.index.values))
        
        # Dataset specific calculations
        if self.name == 'shean':
            # Load all data
            ds_all = pd.read_csv(self.ds_fp + self.ds_fn)            
            ds_all['RegO1'] = ds_all[self.rgi_glacno_cn].values.astype(int)
            # Select data for specific region
            ds_reg = ds_all[ds_all['RegO1']==self.rgi_regionO1].copy()
            ds_reg.reset_index(drop=True, inplace=True)
            # Glacier number and index for comparison
            ds_reg['glacno'] = ((ds_reg[self.rgi_glacno_cn] % 1) * 10**5).round(0).astype(int)
            ds_reg['RGIId'] = ('RGI60-' + str(self.rgi_regionO1) + '.' + 
                               (ds_reg['glacno'] / 10**5).apply(lambda x: '%.5f' % x).astype(str).str.split('.').str[1])
            # Select glaciers with mass balance data
            ds = (ds_reg.iloc[np.where(ds_reg['glacno'].isin(main_glac_rgi['glacno']) == True)[0],:]
                  ).copy()
            ds.reset_index(drop=True, inplace=True)
            # Elevation indices
            elev_bins = main_glac_hyps.columns.values.astype(int)
            elev_bin_interval = elev_bins[1] - elev_bins[0]
            ds['z1_idx'] = (
                    (main_glac_hyps.iloc[ds['glacno'].map(glacnodict)].values != 0).argmax(axis=1).astype(int))
            ds['z2_idx'] = (
                    (main_glac_hyps.iloc[ds['glacno'].map(glacnodict)].values.cumsum(1)).argmax(axis=1).astype(int))
            # Lower and upper bin elevations [masl]
            ds['z1'] = elev_bins[ds['z1_idx'].values] - elev_bin_interval/2
            ds['z2'] = elev_bins[ds['z2_idx'].values] + elev_bin_interval/2
            # Area [km2]
            ds['area_km2'] = np.nan
            for x in range(ds.shape[0]):
                ds.loc[x,'area_km2'] = (
                        main_glac_hyps.iloc[glacnodict[ds.loc[x,'glacno']], 
                                            ds.loc[x,'z1_idx']:ds.loc[x,'z2_idx']+1].sum())
            # Time indices
            ds['t1'] = ds[self.t1_cn]
            ds['t2'] = ds[self.t2_cn]
            ds['t1_year'] = ds['t1'].astype(int)
            ds['t1_month'] = round(ds['t1'] % ds['t1_year'] * 12 + 1)
            #  add 1 to account for the fact that January starts with value of 1
#            ds['t1_month'] = (ds['t1'] % ds['t1_year'] * 12 + 1).astype(int)
#            ds['t1_daysinmonth'] = ds.apply(lambda row: modelsetup.daysinmonth(row['t1_year'], row['t1_month']), axis=1)
#            ds['t1_day'] = ((ds['t1'] % ds['t1_year'] * 12 + 1) % 1 * ds['t1_daysinmonth']).astype(int)
            ds['t2_year'] = ds['t2'].astype(int)
            ds['t2_month'] = round(ds['t2'] % ds['t2_year'] * 12)
             #  do not need to add one for t2 because we want the last full time step
#            ds['t2_month'] = (ds['t2'] % ds['t2_year'] * 12 + 1).astype(int)
#            ds['t2_daysinmonth'] = ds.apply(lambda row: modelsetup.daysinmonth(row['t2_year'], row['t2_month']), axis=1)
#            ds['t2_day'] = ((ds['t2'] % ds['t2_year'] * 12 + 1) % 1 * ds['t2_daysinmonth']).astype(int)
#            ds['t1_datetime'] = pd.to_datetime(
#                    pd.DataFrame({'year':ds.t1_year.values, 'month':ds.t1_month.values, 'day':ds.t1_day.values}))
#            ds['t2_datetime'] = pd.to_datetime(
#                    pd.DataFrame({'year':ds.t2_year.values, 'month':ds.t2_month.values, 'day':ds.t2_day.values}))
            # Remove data with dates outside of calibration period
            year_decimal_min = dates_table.loc[0,'year'] + dates_table.loc[0,'month'] / 12
            year_decimal_max = (dates_table.loc[dates_table.shape[0]-1,'year'] + 
                                (dates_table.loc[dates_table.shape[0]-1,'month'] + 1) / 12)
            ds = ds[ds['t1_year'] + ds['t1_month'] / 12 >= year_decimal_min]
            ds = ds[ds['t2_year'] + ds['t2_month'] / 12 <= year_decimal_max]
            ds.reset_index(drop=True, inplace=True)    
            
            # Determine time indices (exclude spinup years, since massbal fxn discards spinup years)
            ds['t1_idx'] = np.nan
            ds['t2_idx'] = np.nan
            for x in range(ds.shape[0]):
                ds.loc[x,'t1_idx'] = (dates_table[(ds.loc[x, 't1_year'] == dates_table['year']) & 
                                                  (ds.loc[x, 't1_month'] == dates_table['month'])].index.values[0])
                ds.loc[x,'t2_idx'] = (dates_table[(ds.loc[x, 't2_year'] == dates_table['year']) & 
                                                  (ds.loc[x, 't2_month'] == dates_table['month'])].index.values[0])
            ds['t1_idx'] = ds['t1_idx'].astype(int)
            # Specific mass balance [mwea]
            ds['mb_mwe'] = ds[self.mb_mwea_cn] * (ds['t2'] - ds['t1'])
            ds['mb_mwe_err'] = ds[self.mb_mwea_err_cn] * (ds['t2'] - ds['t1']) 
#            # Total mass change [Gt]
#            ds['mb_gt'] = ds[self.mb_vol_cn] * (ds['t2'] - ds['t1']) * (1/1000)**3 * input.density_water / 1000
#            ds['mb_gt_err'] = ds[self.mb_vol_err_cn] * (ds['t2'] - ds['t1']) * (1/1000)**3 * input.density_water / 1000
            # Observation type
            ds['obs_type'] = 'mb_geo'
            # Add columns with nan for things not in list
            ds_addcols = [x for x in ds_output_cols if x not in ds.columns.values]
            for colname in ds_addcols:
                ds[colname] = np.nan
#            ds['group_name'] = np.nan
#            ds['sla_m'] = np.nan
#            ds['WGMS_ID'] = np.nan
            
        elif self.name == 'brun':
            print('code brun')
            
        elif self.name == 'mauer':
            # Load all data
            ds_all = pd.read_csv(self.ds_fp + self.ds_fn)            
            ds_all['RegO1'] = ds_all[self.rgi_glacno_cn].values.astype(int)
            # Select data for specific region
            ds_reg = ds_all[ds_all['RegO1']==self.rgi_regionO1].copy()
            ds_reg.reset_index(drop=True, inplace=True)
            # Glacier number and index for comparison
            ds_reg['glacno'] = ((ds_reg[self.rgi_glacno_cn] % 1) * 10**5).round(0).astype(int)
            ds_reg['RGIId'] = ('RGI60-' + str(self.rgi_regionO1) + '.' + 
                               (ds_reg['glacno'] / 10**5).apply(lambda x: '%.5f' % x).astype(str).str.split('.').str[1])
            # Select glaciers with mass balance data
            ds = (ds_reg.iloc[np.where(ds_reg['glacno'].isin(main_glac_rgi['glacno']) == True)[0],:]
                  ).copy()
            ds.reset_index(drop=True, inplace=True)
            # Elevation indices
            elev_bins = main_glac_hyps.columns.values.astype(int)
            elev_bin_interval = elev_bins[1] - elev_bins[0]
            ds['z1_idx'] = (
                    (main_glac_hyps.iloc[ds['glacno'].map(glacnodict)].values != 0).argmax(axis=1).astype(int))
            ds['z2_idx'] = (
                    (main_glac_hyps.iloc[ds['glacno'].map(glacnodict)].values.cumsum(1)).argmax(axis=1).astype(int))
            # Lower and upper bin elevations [masl]
            ds['z1'] = elev_bins[ds['z1_idx'].values] - elev_bin_interval/2
            ds['z2'] = elev_bins[ds['z2_idx'].values] + elev_bin_interval/2
            # Area [km2]
            ds['area_km2'] = np.nan
            for x in range(ds.shape[0]):
                ds.loc[x,'area_km2'] = (
                        main_glac_hyps.iloc[glacnodict[ds.loc[x,'glacno']], 
                                            ds.loc[x,'z1_idx']:ds.loc[x,'z2_idx']+1].sum())
            # Time indices
            ds['t1'] = ds[self.t1_cn]
            ds['t2'] = ds[self.t2_cn]
            ds['t1_year'] = ds['t1'].astype(int)
            ds['t1_month'] = round(ds['t1'] % ds['t1_year'] * 12 + 1)
            #  add 1 to account for the fact that January starts with value of 1
            ds.loc[ds['t1_month'] > 12, 't1_month'] = 12
            ds['t2_year'] = ds['t2'].astype(int)
            ds['t2_month'] = 2
            # Remove data with dates outside of calibration period
            year_decimal_min = dates_table.loc[0,'year'] + dates_table.loc[0,'month'] / 12
            year_decimal_max = (dates_table.loc[dates_table.shape[0]-1,'year'] + 
                                (dates_table.loc[dates_table.shape[0]-1,'month'] + 1) / 12)
            ds = ds[ds['t1_year'] + ds['t1_month'] / 12 >= year_decimal_min]
            ds = ds[ds['t2_year'] + ds['t2_month'] / 12 <= year_decimal_max]
            ds.reset_index(drop=True, inplace=True)                
            # Determine time indices (exclude spinup years, since massbal fxn discards spinup years)
            ds['t1_idx'] = np.nan
            ds['t2_idx'] = np.nan
            for x in range(ds.shape[0]):
                ds.loc[x,'t1_idx'] = (dates_table[(ds.loc[x, 't1_year'] == dates_table['year']) & 
                                                  (ds.loc[x, 't1_month'] == dates_table['month'])].index.values[0])
                ds.loc[x,'t2_idx'] = (dates_table[(ds.loc[x, 't2_year'] == dates_table['year']) & 
                                                  (ds.loc[x, 't2_month'] == dates_table['month'])].index.values[0])
            ds['t1_idx'] = ds['t1_idx'].astype(int)
            # Specific mass balance [mwea]
            ds['mb_mwe'] = ds[self.mb_mwea_cn] * (ds['t2'] - ds['t1'])
            ds['mb_mwe_err'] = ds[self.mb_mwea_err_cn] * (ds['t2'] - ds['t1']) 
            # Observation type
            ds['obs_type'] = 'mb_geo'
            # Add columns with nan for things not in list
            ds_addcols = [x for x in ds_output_cols if x not in ds.columns.values]
            for colname in ds_addcols:
                ds[colname] = np.nan
        
        elif self.name == 'wgms_d':
            # Load all data
            ds_all = pd.read_csv(self.ds_fp + self.ds_fn, encoding='latin1')
            ds_all['RegO1'] = ds_all[self.rgi_glacno_cn].values.astype(int)
            # Select data for specific region
            ds_reg = ds_all[ds_all['RegO1']==self.rgi_regionO1].copy()
            # Glacier number and index for comparison
            ds_reg['glacno'] = ((ds_reg[self.rgi_glacno_cn] % 1) * 10**5).round(0).astype(int)
            # Select glaciers from those being modeled using main_glac_rgi
            ds = (ds_reg.iloc[np.where(ds_reg['glacno'].isin(main_glac_rgi[input.rgi_O1Id_colname]) == True)[0],:]
                  ).copy()
            ds.reset_index(drop=True, inplace=True)
            # Elevation indices
            elev_bins = main_glac_hyps.columns.values.astype(int)
            elev_bin_interval = elev_bins[1] - elev_bins[0]
            ds['z1_idx'] = np.nan
            ds['z2_idx'] = np.nan
            ds.loc[ds[self.z1_cn] == 9999, 'z1_idx'] = (
                    (main_glac_hyps.iloc[ds.loc[ds[self.z1_cn] == 9999, 'glacno'].map(glacnodict)].values != 0)
                     .argmax(axis=1))
            ds.loc[ds[self.z2_cn] == 9999, 'z2_idx'] = (
                    (main_glac_hyps.iloc[ds.loc[ds[self.z2_cn] == 9999, 'glacno'].map(glacnodict)].values.cumsum(1))
                     .argmax(axis=1))
            ds.loc[ds[self.z1_cn] != 9999, 'z1_idx'] = (
                    ((np.tile(elev_bins, (ds.loc[ds[self.z1_cn] != 9999, self.z1_cn].shape[0],1)) - 
                      ds.loc[ds[self.z1_cn] != 9999, self.z1_cn][:,np.newaxis]) > 0).argmax(axis=1))
            ds.loc[ds[self.z2_cn] != 9999, 'z2_idx'] = (
                    ((np.tile(elev_bins, (ds.loc[ds[self.z2_cn] != 9999, self.z2_cn].shape[0],1)) - 
                      ds.loc[ds[self.z2_cn] != 9999, self.z2_cn][:,np.newaxis]) > 0).argmax(axis=1) - 1)
            ds['z1_idx'] = ds['z1_idx'].values.astype(int)
            ds['z2_idx'] = ds['z2_idx'].values.astype(int)
            # Lower and upper bin elevations [masl]
            ds['z1'] = elev_bins[ds['z1_idx'].values] - elev_bin_interval/2
            ds['z2'] = elev_bins[ds['z2_idx'].values] + elev_bin_interval/2
            # Area [km2]
            #  use WGMS area when provided; otherwise use area from RGI
            ds['area_km2_rgi'] = np.nan
            for x in range(ds.shape[0]):
                ds.loc[x,'area_km2_rgi'] = (
                        main_glac_hyps.iloc[glacnodict[ds.loc[x,'glacno']], 
                                            ds.loc[x,'z1_idx']:ds.loc[x,'z2_idx']+1].sum())            
            ds['area_km2'] = np.nan
            ds.loc[ds.AREA_SURVEY_YEAR.isnull(), 'area_km2'] = ds.loc[ds.AREA_SURVEY_YEAR.isnull(), 'area_km2_rgi']
            ds.loc[ds.AREA_SURVEY_YEAR.notnull(), 'area_km2'] = ds.loc[ds.AREA_SURVEY_YEAR.notnull(), 
                                                                       'AREA_SURVEY_YEAR']
            # Time indices
            # remove data that does not have reference date or survey data
            ds = ds[np.isnan(ds['REFERENCE_DATE']) == False]
            ds = ds[np.isnan(ds['SURVEY_DATE']) == False]
            ds.reset_index(drop=True, inplace=True)
            # Extract date information
            ds['t1_year'] = ds['REFERENCE_DATE'].astype(str).str.split('.').str[0].str[:4].astype(int)
            ds['t1_month'] = ds['REFERENCE_DATE'].astype(str).str.split('.').str[0].str[4:6].astype(int)
            ds['t1_day'] = ds['REFERENCE_DATE'].astype(str).str.split('.').str[0].str[6:].astype(int)
            ds['t2_year'] = ds['SURVEY_DATE'].astype(str).str.split('.').str[0].str[:4].astype(int)
            ds['t2_month'] = ds['SURVEY_DATE'].astype(str).str.split('.').str[0].str[4:6].astype(int)
            ds['t2_day'] = ds['SURVEY_DATE'].astype(str).str.split('.').str[0].str[6:].astype(int)
            # if month/day unknown for start or end period, then replace with water year
            # Add latitude 
            latdict = dict(zip(main_glac_rgi['RGIId'], main_glac_rgi['CenLat']))
            ds['CenLat'] = ds['RGIId'].map(latdict)
            ds['lat_category'] = np.nan
            ds.loc[ds['CenLat'] >= input.lat_threshold, 'lat_category'] = 'northernmost'
            ds.loc[(ds['CenLat'] < input.lat_threshold) & (ds['CenLat'] > 0), 'lat_category'] = 'north'
            ds.loc[(ds['CenLat'] <= 0) & (ds['CenLat'] > -1*input.lat_threshold), 'lat_category'] = 'south'
            ds.loc[ds['CenLat'] <= -1*input.lat_threshold, 'lat_category'] = 'southernmost'
            ds['months_wintersummer'] = ds['lat_category'].map(input.monthdict)
            ds['winter_begin'] = ds['months_wintersummer'].apply(lambda x: x[0])
            ds['winter_end'] = ds['months_wintersummer'].apply(lambda x: x[1])
            ds['summer_begin'] = ds['months_wintersummer'].apply(lambda x: x[2])
            ds['summer_end'] = ds['months_wintersummer'].apply(lambda x: x[3])
            ds.loc[ds['t1_month'] == 99, 't1_month'] = ds.loc[ds['t1_month'] == 99, 'winter_begin']
            ds.loc[ds['t1_day'] == 99, 't1_day'] = 1
            ds.loc[ds['t2_month'] == 99, 't2_month'] = ds.loc[ds['t2_month'] == 99, 'winter_begin'] - 1
            for x in range(ds.shape[0]):
                if ds.loc[x, 't2_day'] == 99:
                    try:
                        ds.loc[x, 't2_day'] = (
                                dates_table.loc[(ds.loc[x, 't2_year'] == dates_table['year']) & 
                                                (ds.loc[x, 't2_month'] == dates_table['month']), 'daysinmonth']
                                                .values[0])
                    except:
                        ds.loc[x, 't2_day'] = 28    
            # Replace poor values of months
            ds['t1_month'] = ds['t1_month'].map(lambda x: x if x <=12 else x%12)
            ds['t2_month'] = ds['t2_month'].map(lambda x: x if x <=12 else x%12)
            # Replace poor values of days
            ds['t1_daysinmonth'] = (
                    [calendar.monthrange(ds.loc[x,'t1_year'], ds.loc[x,'t1_month'])[1] for x in range(ds.shape[0])])
            ds['t2_daysinmonth'] = (
                    [calendar.monthrange(ds.loc[x,'t2_year'], ds.loc[x,'t2_month'])[1] for x in range(ds.shape[0])])
            ds['t1_day'] = (ds.apply(lambda x: x['t1_day'] if x['t1_day'] <= x['t1_daysinmonth'] 
                                     else x['t1_daysinmonth'], axis=1))
            ds['t2_day'] = (ds.apply(lambda x: x['t2_day'] if x['t2_day'] <= x['t2_daysinmonth'] 
                                     else x['t2_daysinmonth'], axis=1))
            # Calculate decimal year and drop measurements outside of calibration period
            ds['t1_datetime'] = pd.to_datetime(
                    pd.DataFrame({'year':ds.t1_year.values, 'month':ds.t1_month.values, 'day':ds.t1_day.values}))
            ds['t2_datetime'] = pd.to_datetime(
                    pd.DataFrame({'year':ds.t2_year.values, 'month':ds.t2_month.values, 'day':ds.t2_day.values}))
            ds['t1_doy'] = ds.t1_datetime.dt.strftime("%j").astype(float)
            ds['t2_doy'] = ds.t2_datetime.dt.strftime("%j").astype(float)
            ds['t1_daysinyear'] = (
                    (pd.to_datetime(pd.DataFrame({'year':ds.t1_year.values, 'month':12, 'day':31})) - 
                     pd.to_datetime(pd.DataFrame({'year':ds.t1_year.values, 'month':1, 'day':1}))).dt.days + 1)
            ds['t2_daysinyear'] = (
                    (pd.to_datetime(pd.DataFrame({'year':ds.t2_year.values, 'month':12, 'day':31})) - 
                     pd.to_datetime(pd.DataFrame({'year':ds.t2_year.values, 'month':1, 'day':1}))).dt.days + 1)
            ds['t1'] = ds.t1_year + ds.t1_doy / ds.t1_daysinyear
            ds['t2'] = ds.t2_year + ds.t2_doy / ds.t2_daysinyear
            end_datestable = dates_table.loc[dates_table.shape[0]-1, 'date']
            end_datetime = datetime.datetime(end_datestable.year, end_datestable.month + 1, end_datestable.day)
            ds = ds[ds['t1_datetime'] >= dates_table.loc[0, 'date']]
            ds = ds[ds['t2_datetime'] < end_datetime]
            ds.reset_index(drop=True, inplace=True)
            # Time indices
            #  exclude spinup years, since massbal fxn discards spinup years
            ds['t1_idx'] = np.nan
            ds['t2_idx'] = np.nan
            for x in range(ds.shape[0]):
                ds.loc[x,'t1_idx'] = (dates_table[(ds.loc[x, 't1_year'] == dates_table['year']) & 
                                                  (ds.loc[x, 't1_month'] == dates_table['month'])].index.values[0])
                ds.loc[x,'t2_idx'] = (dates_table[(ds.loc[x, 't2_year'] == dates_table['year']) & 
                                                  (ds.loc[x, 't2_month'] == dates_table['month'])].index.values[0])
            # Specific mass balance [mwe]
            #  if thickness change is available, then compute the specific mass balance with the thickness change
            #  otherwise, use the volume change and area to estimate the specific mass balance
            # using thickness change
            ds['mb_mwe'] = ds[self.thickness_chg_cn] / 1000 * input.density_ice / input.density_water
            ds['mb_mwe_err'] = ds[self.thickness_chg_err_cn] / 1000 * input.density_ice / input.density_water
            # using volume change (note: units volume change [1000 m3] and area [km2])
            ds.loc[ds.mb_mwe.isnull(), 'mb_mwe'] = (
                    ds.loc[ds.mb_mwe.isnull(), self.volume_chg_cn] * 1000 / ds.loc[ds.mb_mwe.isnull(), 'area_km2'] * 
                    (1/1000)**2 * input.density_ice / input.density_water)
            ds.loc[ds.mb_mwe.isnull(), 'mb_mwe'] = (
                    ds.loc[ds.mb_mwe.isnull(), self.volume_chg_err_cn] * 1000 / ds.loc[ds.mb_mwe.isnull(), 'area_km2'] * 
                    (1/1000)**2 * input.density_ice / input.density_water)
            # Observation type
            ds['obs_type'] = 'mb_geo'
            ds['group_name'] = np.nan
            ds['sla_m'] = np.nan
        elif self.name == 'wgms_ee':
            # Load all data
            ds_all = pd.read_csv(self.ds_fp + self.ds_fn, encoding='latin1')            
            ds_all['RegO1'] = ds_all[self.rgi_glacno_cn].values.astype(int)
            # Select data for specific region
            ds_reg = ds_all[ds_all['RegO1']==self.rgi_regionO1].copy()
            # Glacier number and index for comparison
            ds_reg['glacno'] = ((ds_reg[self.rgi_glacno_cn] % 1) * 10**5).round(0).astype(int)
            # Fill in glaciers without reference data
            ds_reg.loc[ds_reg.BEGIN_PERIOD.isnull(), 'BEGIN_PERIOD'] = (
                    (ds_reg.loc[ds_reg.BEGIN_PERIOD.isnull(), 'YEAR'] - 1) * 10**4 + 9999)
            ds_reg.loc[ds_reg.END_PERIOD.isnull(), 'END_PERIOD'] = (
                    ds_reg.loc[ds_reg.END_PERIOD.isnull(), 'YEAR'] * 10**4 + 9999)
            # Select glaciers from those being modeled using main_glac_rgi
            ds = (ds_reg.iloc[np.where(ds_reg['glacno'].isin(main_glac_rgi[input.rgi_O1Id_colname]) == True)[0],:]
                  ).copy()
            ds.reset_index(drop=True, inplace=True)
            # Elevation indices
            elev_bins = main_glac_hyps.columns.values.astype(int)
            elev_bin_interval = elev_bins[1] - elev_bins[0]
            ds['z1_idx'] = np.nan
            ds['z2_idx'] = np.nan
            ds.loc[ds[self.z1_cn] == 9999, 'z1_idx'] = (
                    (main_glac_hyps.iloc[ds.loc[ds[self.z1_cn] == 9999, 'glacno'].map(glacnodict)].values != 0)
                     .argmax(axis=1))
            ds.loc[ds[self.z2_cn] == 9999, 'z2_idx'] = (
                    (main_glac_hyps.iloc[ds.loc[ds[self.z2_cn] == 9999, 'glacno'].map(glacnodict)].values.cumsum(1))
                     .argmax(axis=1))
            ds.loc[ds[self.z1_cn] != 9999, 'z1_idx'] = (
                    ((np.tile(elev_bins, (ds.loc[ds[self.z1_cn] != 9999, self.z1_cn].shape[0],1)) - 
                      ds.loc[ds[self.z1_cn] != 9999, self.z1_cn][:,np.newaxis]) > 0).argmax(axis=1))
            ds.loc[ds[self.z2_cn] != 9999, 'z2_idx'] = (
                    ((np.tile(elev_bins, (ds.loc[ds[self.z2_cn] != 9999, self.z2_cn].shape[0],1)) - 
                      ds.loc[ds[self.z2_cn] != 9999, self.z2_cn][:,np.newaxis]) > 0).argmax(axis=1) - 1)
            ds['z1_idx'] = ds['z1_idx'].values.astype(int)
            ds['z2_idx'] = ds['z2_idx'].values.astype(int)
            # Lower and upper bin elevations [masl]
            ds['z1'] = elev_bins[ds['z1_idx'].values] - elev_bin_interval/2
            ds['z2'] = elev_bins[ds['z2_idx'].values] + elev_bin_interval/2
            # Area [km2]
            ds['area_km2'] = np.nan
            for x in range(ds.shape[0]):
                ds.loc[x,'area_km2'] = (
                        main_glac_hyps.iloc[glacnodict[ds.loc[x,'glacno']], 
                                            ds.loc[x,'z1_idx']:ds.loc[x,'z2_idx']+1].sum())
            ds = ds[ds['area_km2'] > 0]
            ds.reset_index(drop=True, inplace=True)
            # Time indices
            #  winter and summer balances typically have the same data for 'BEGIN_PERIOD' and 'END_PERIOD' as the annual
            #  measurements, so need to set these dates manually
            ds['t1_year'] = ds['BEGIN_PERIOD'].astype(str).str.split('.').str[0].str[:4].astype(int)
            ds['t1_month'] = ds['BEGIN_PERIOD'].astype(str).str.split('.').str[0].str[4:6].astype(int)
            ds['t1_day'] = ds['BEGIN_PERIOD'].astype(str).str.split('.').str[0].str[6:].astype(int)
            ds['t2_year'] = ds['END_PERIOD'].astype(str).str.split('.').str[0].str[:4].astype(int)
            ds['t2_month'] = ds['END_PERIOD'].astype(str).str.split('.').str[0].str[4:6].astype(int)
            ds['t2_day'] = ds['END_PERIOD'].astype(str).str.split('.').str[0].str[6:].astype(int)            
            # if annual measurement and month/day unknown for start or end period, then replace with water year
            # Add latitude 
            latdict = dict(zip(main_glac_rgi['RGIId'], main_glac_rgi['CenLat']))
            ds['CenLat'] = ds['RGIId'].map(latdict)
            ds['lat_category'] = np.nan
            ds.loc[ds['CenLat'] >= input.lat_threshold, 'lat_category'] = 'northernmost'
            ds.loc[(ds['CenLat'] < input.lat_threshold) & (ds['CenLat'] > 0), 'lat_category'] = 'north'
            ds.loc[(ds['CenLat'] <= 0) & (ds['CenLat'] > -1*input.lat_threshold), 'lat_category'] = 'south'
            ds.loc[ds['CenLat'] <= -1*input.lat_threshold, 'lat_category'] = 'southernmost'
            ds['months_wintersummer'] = ds['lat_category'].map(input.monthdict)
            ds['winter_begin'] = ds['months_wintersummer'].apply(lambda x: x[0])
            ds['winter_end'] = ds['months_wintersummer'].apply(lambda x: x[1])
            ds['summer_begin'] = ds['months_wintersummer'].apply(lambda x: x[2])
            ds['summer_end'] = ds['months_wintersummer'].apply(lambda x: x[3])
            # annual start
            ds.loc[ds['t1_month'] == 99, 't1_month'] = ds.loc[ds['t1_month'] == 99, 'winter_begin']
            ds.loc[ds['t1_day'] == 99, 't1_day'] = 1
            ds.loc[ds['t2_month'] == 99, 't2_month'] = ds.loc[ds['t2_month'] == 99, 'winter_begin'] - 1
            for x in range(ds.shape[0]):
                if ds.loc[x, 't2_day'] == 99:
                    try:
                        ds.loc[x, 't2_day'] = (
                                dates_table.loc[(ds.loc[x, 't2_year'] == dates_table['year']) & 
                                                (ds.loc[x, 't2_month'] == dates_table['month']), 'daysinmonth']
                                                .values[0])
                    except:
                        ds.loc[x, 't2_day'] = 28
            # If period is summer/winter, adjust dates accordingly
            for x in range(ds.shape[0]):
                if (((ds.loc[x, 'lat_category'] == 'north') or (ds.loc[x, 'lat_category'] == 'northern')) and 
                    (ds.loc[x, 'period'] == 'summer')):
                    ds.loc[x, 't1_year'] = ds.loc[x, 't1_year'] + 1
                    ds.loc[x, 't1_month'] = ds.loc[x, 'summer_begin']
                    ds.loc[x, 't2_month'] = ds.loc[x, 'summer_end']
                elif (((ds.loc[x, 'lat_category'] == 'south') or (ds.loc[x, 'lat_category'] == 'southernmost')) and 
                    (ds.loc[x, 'period'] == 'summer')):
                    ds.loc[x, 't1_month'] = ds.loc[x, 'summer_begin']
                    ds.loc[x, 't2_month'] = ds.loc[x, 'summer_end']
                elif (((ds.loc[x, 'lat_category'] == 'north') or (ds.loc[x, 'lat_category'] == 'northern')) and 
                    (ds.loc[x, 'period'] == 'winter')):
                    ds.loc[x, 't1_month'] = ds.loc[x, 'winter_begin']
                    ds.loc[x, 't2_month'] = ds.loc[x, 'winter_end']
                elif (((ds.loc[x, 'lat_category'] == 'south') or (ds.loc[x, 'lat_category'] == 'southernmost')) and 
                    (ds.loc[x, 'period'] == 'summer')):
                    ds.loc[x, 't1_year'] = ds.loc[x, 't1_year'] + 1
                    ds.loc[x, 't1_month'] = ds.loc[x, 'winter_begin']
                    ds.loc[x, 't2_month'] = ds.loc[x, 'winter_end']
                ds.loc[x, 't1_day'] = 1
                ds.loc[x, 't2_day'] = calendar.monthrange(ds.loc[x, 't2_year'], ds.loc[x, 't2_month'])[1]
            # Replace poor values of months
            ds['t1_month'] = ds['t1_month'].map(lambda x: x if x <=12 else x%12)
            ds['t2_month'] = ds['t2_month'].map(lambda x: x if x <=12 else x%12)
            # Calculate decimal year and drop measurements outside of calibration period
            ds['t1_datetime'] = pd.to_datetime(
                    pd.DataFrame({'year':ds.t1_year.values, 'month':ds.t1_month.values, 'day':ds.t1_day.values}))
            ds['t2_datetime'] = pd.to_datetime(
                    pd.DataFrame({'year':ds.t2_year.values, 'month':ds.t2_month.values, 'day':ds.t2_day.values}))
            ds['t1_doy'] = ds.t1_datetime.dt.strftime("%j").astype(float)
            ds['t2_doy'] = ds.t2_datetime.dt.strftime("%j").astype(float)
            ds['t1_daysinyear'] = (
                    (pd.to_datetime(pd.DataFrame({'year':ds.t1_year.values, 'month':12, 'day':31})) - 
                     pd.to_datetime(pd.DataFrame({'year':ds.t1_year.values, 'month':1, 'day':1}))).dt.days + 1)
            ds['t2_daysinyear'] = (
                    (pd.to_datetime(pd.DataFrame({'year':ds.t2_year.values, 'month':12, 'day':31})) - 
                     pd.to_datetime(pd.DataFrame({'year':ds.t2_year.values, 'month':1, 'day':1}))).dt.days + 1)
            ds['t1'] = ds.t1_year + ds.t1_doy / ds.t1_daysinyear
            ds['t2'] = ds.t2_year + ds.t2_doy / ds.t2_daysinyear
            end_datestable = dates_table.loc[dates_table.shape[0]-1, 'date']
            end_datetime = datetime.datetime(end_datestable.year, end_datestable.month + 1, end_datestable.day)
            ds = ds[ds['t1_datetime'] >= dates_table.loc[0, 'date']]
            ds = ds[ds['t2_datetime'] < end_datetime]
            ds.reset_index(drop=True, inplace=True)
            # Annual, summer, and winter time indices
            #  exclude spinup years, since massbal fxn discards spinup years
            ds['t1_idx'] = np.nan
            ds['t2_idx'] = np.nan
            for x in range(ds.shape[0]):
                ds.loc[x,'t1_idx'] = (dates_table[(ds.loc[x, 't1_year'] == dates_table['year']) & 
                                                  (ds.loc[x, 't1_month'] == dates_table['month'])].index.values[0])
                ds.loc[x,'t2_idx'] = (dates_table[(ds.loc[x, 't2_year'] == dates_table['year']) & 
                                                  (ds.loc[x, 't2_month'] == dates_table['month'])].index.values[0])
            # Specific mass balance [mwe]
            ds['mb_mwe'] = ds[self.mb_mwe_cn] / 1000
            ds['mb_mwe_err'] = ds[self.mb_mwe_err_cn] / 1000
#            # Total mass change [Gt]
#            ds['mb_gt'] = ds[self.mb_mwe_cn] / 1000 * ds['area_km2'] * 1000**2 * input.density_water / 1000 / 10**9
#            ds['mb_gt_err'] = (ds[self.mb_mwe_err_cn] / 1000 * ds['area_km2'] * 1000**2 * input.density_water / 1000 
#                               / 10**9)
            # Observation type
            ds['obs_type'] = 'mb_glac'
            ds['group_name'] = np.nan
            ds['sla_m'] = np.nan
            
        elif self.name == 'cogley':
            # Load all data
            ds_all = pd.read_csv(self.ds_fp + self.ds_fn, encoding='latin1')
            ds_all['RegO1'] = ds_all[self.rgi_glacno_cn].values.astype(int)
            # Select data for specific region
            ds_reg = ds_all[ds_all['RegO1']==self.rgi_regionO1].copy()
            # Glacier number and index for comparison
            ds_reg['glacno'] = ((ds_reg[self.rgi_glacno_cn] % 1) * 10**5).round(0).astype(int)
            # Select glaciers from those being modeled using main_glac_rgi
            ds = (ds_reg.iloc[np.where(ds_reg['glacno'].isin(main_glac_rgi[input.rgi_O1Id_colname]) == True)[0],:]
                  ).copy()
            ds.reset_index(drop=True, inplace=True)
            # Elevation indices
            elev_bins = main_glac_hyps.columns.values.astype(int)
            elev_bin_interval = elev_bins[1] - elev_bins[0]
            ds['z1_idx'] = np.nan
            ds['z2_idx'] = np.nan
            ds.loc[ds[self.z1_cn] == 9999, 'z1_idx'] = (
                    (main_glac_hyps.iloc[ds.loc[ds[self.z1_cn] == 9999, 'glacno'].map(glacnodict)].values != 0)
                     .argmax(axis=1))
            ds.loc[ds[self.z2_cn] == 9999, 'z2_idx'] = (
                    (main_glac_hyps.iloc[ds.loc[ds[self.z2_cn] == 9999, 'glacno'].map(glacnodict)].values.cumsum(1))
                     .argmax(axis=1))
            ds.loc[ds[self.z1_cn] != 9999, 'z1_idx'] = (
                    ((np.tile(elev_bins, (ds.loc[ds[self.z1_cn] != 9999, self.z1_cn].shape[0],1)) - 
                      ds.loc[ds[self.z1_cn] != 9999, self.z1_cn][:,np.newaxis]) > 0).argmax(axis=1))
            ds.loc[ds[self.z2_cn] != 9999, 'z2_idx'] = (
                    ((np.tile(elev_bins, (ds.loc[ds[self.z2_cn] != 9999, self.z2_cn].shape[0],1)) - 
                      ds.loc[ds[self.z2_cn] != 9999, self.z2_cn][:,np.newaxis]) > 0).argmax(axis=1) - 1)
            ds['z1_idx'] = ds['z1_idx'].values.astype(int)
            ds['z2_idx'] = ds['z2_idx'].values.astype(int)
            # Lower and upper bin elevations [masl]
            ds['z1'] = elev_bins[ds['z1_idx'].values] - elev_bin_interval/2
            ds['z2'] = elev_bins[ds['z2_idx'].values] + elev_bin_interval/2
            # Area [km2]
            #  use WGMS area when provided; otherwise use area from RGI
            ds['area_km2_rgi'] = np.nan
            for x in range(ds.shape[0]):
                ds.loc[x,'area_km2_rgi'] = (
                        main_glac_hyps.iloc[glacnodict[ds.loc[x,'glacno']], 
                                            ds.loc[x,'z1_idx']:ds.loc[x,'z2_idx']+1].sum())
            # Time indices
            ds['t1_year'] = ds['REFERENCE_DATE'].astype(str).str.split('.').str[0].str[:4].astype(int)
            ds['t1_month'] = ds['REFERENCE_DATE'].astype(str).str.split('.').str[0].str[4:6].astype(int)
            ds['t1_day'] = ds['REFERENCE_DATE'].astype(str).str.split('.').str[0].str[6:].astype(int)
            ds['t2_year'] = ds['SURVEY_DATE'].astype(str).str.split('.').str[0].str[:4].astype(int)
            ds['t2_month'] = ds['SURVEY_DATE'].astype(str).str.split('.').str[0].str[4:6].astype(int)
            ds['t2_day'] = ds['SURVEY_DATE'].astype(str).str.split('.').str[0].str[6:].astype(int)
            # if month/day unknown for start or end period, then replace with water year
            # Add latitude 
            latdict = dict(zip(main_glac_rgi['RGIId'], main_glac_rgi['CenLat']))
            ds['CenLat'] = ds['RGIId'].map(latdict)
            ds['lat_category'] = np.nan
            ds.loc[ds['CenLat'] >= input.lat_threshold, 'lat_category'] = 'northernmost'
            ds.loc[(ds['CenLat'] < input.lat_threshold) & (ds['CenLat'] > 0), 'lat_category'] = 'north'
            ds.loc[(ds['CenLat'] <= 0) & (ds['CenLat'] > -1*input.lat_threshold), 'lat_category'] = 'south'
            ds.loc[ds['CenLat'] <= -1*input.lat_threshold, 'lat_category'] = 'southernmost'
            ds['months_wintersummer'] = ds['lat_category'].map(input.monthdict)
            ds['winter_begin'] = ds['months_wintersummer'].apply(lambda x: x[0])
            ds['winter_end'] = ds['months_wintersummer'].apply(lambda x: x[1])
            ds['summer_begin'] = ds['months_wintersummer'].apply(lambda x: x[2])
            ds['summer_end'] = ds['months_wintersummer'].apply(lambda x: x[3])
            ds.loc[ds['t1_month'] == 99, 't1_month'] = ds.loc[ds['t1_month'] == 99, 'winter_begin']
            ds.loc[ds['t1_day'] == 99, 't1_day'] = 1
            ds.loc[ds['t2_month'] == 99, 't2_month'] = ds.loc[ds['t2_month'] == 99, 'winter_begin'] - 1
            for x in range(ds.shape[0]):
                if ds.loc[x, 't2_day'] == 99:
                    try:
                        ds.loc[x, 't2_day'] = (
                                dates_table.loc[(ds.loc[x, 't2_year'] == dates_table['year']) & 
                                                (ds.loc[x, 't2_month'] == dates_table['month']), 'daysinmonth']
                                                .values[0])
                    except:
                        ds.loc[x, 't2_day'] = 28    
            # Calculate decimal year and drop measurements outside of calibration period
            ds['t1_datetime'] = pd.to_datetime(
                    pd.DataFrame({'year':ds.t1_year.values, 'month':ds.t1_month.values, 'day':ds.t1_day.values}))
            ds['t2_datetime'] = pd.to_datetime(
                    pd.DataFrame({'year':ds.t2_year.values, 'month':ds.t2_month.values, 'day':ds.t2_day.values}))
            ds['t1_doy'] = ds.t1_datetime.dt.strftime("%j").astype(float)
            ds['t2_doy'] = ds.t2_datetime.dt.strftime("%j").astype(float)
            ds['t1_daysinyear'] = (
                    (pd.to_datetime(pd.DataFrame({'year':ds.t1_year.values, 'month':12, 'day':31})) - 
                     pd.to_datetime(pd.DataFrame({'year':ds.t1_year.values, 'month':1, 'day':1}))).dt.days + 1)
            ds['t2_daysinyear'] = (
                    (pd.to_datetime(pd.DataFrame({'year':ds.t2_year.values, 'month':12, 'day':31})) - 
                     pd.to_datetime(pd.DataFrame({'year':ds.t2_year.values, 'month':1, 'day':1}))).dt.days + 1)
            ds['t1'] = ds.t1_year + ds.t1_doy / ds.t1_daysinyear
            ds['t2'] = ds.t2_year + ds.t2_doy / ds.t2_daysinyear
            end_datestable = dates_table.loc[dates_table.shape[0]-1, 'date']
            end_datetime = datetime.datetime(end_datestable.year, end_datestable.month + 1, end_datestable.day)
            ds = ds[ds['t1_datetime'] >= dates_table.loc[0, 'date']]
            ds = ds[ds['t2_datetime'] < end_datetime]
            ds.reset_index(drop=True, inplace=True)
            # Time indices
            #  exclude spinup years, since massbal fxn discards spinup years
            ds['t1_idx'] = np.nan
            ds['t2_idx'] = np.nan
            for x in range(ds.shape[0]):
                ds.loc[x,'t1_idx'] = (dates_table[(ds.loc[x, 't1_year'] == dates_table['year']) & 
                                                  (ds.loc[x, 't1_month'] == dates_table['month'])].index.values[0])
                ds.loc[x,'t2_idx'] = (dates_table[(ds.loc[x, 't2_year'] == dates_table['year']) & 
                                                  (ds.loc[x, 't2_month'] == dates_table['month'])].index.values[0])
            # Specific mass balance [mwe]
            ds['mb_mwe'] = ds[self.mass_chg_cn] / input.density_water * (ds['t2'] - ds['t1'])
            ds['mb_mwe_err'] = ds[self.mass_chg_err_cn] / input.density_water * (ds['t2'] - ds['t1'])
            # Observation type
            ds['obs_type'] = 'mb_geo'
            ds['group_name'] = np.nan
            
        elif self.name == 'group':
            # Load all data
            ds_all = pd.read_csv(self.ds_fp + self.ds_fn, encoding='latin1')
            # Dictionary linking group_names with the RGIIds
            ds_dict_raw = pd.read_csv(self.ds_fp + self.ds_dict_fn)
            ds_dict = dict(zip(ds_dict_raw['RGIId'], ds_dict_raw['group_name']))
            # For each unique group name identify all glaciers associated with the group and test if all those glaciers
            #  are included in the model run via main_glac_rgi
            group_names_unique = list(set(ds_dict.values()))
            ds_dict_keyslist = [[] for x in group_names_unique]
            for n, group in enumerate(group_names_unique):
                ds_dict_keyslist[n] = [group, [k for k, v in ds_dict.items() if v == group]]
                ds_all['glaciers_present'] = set(ds_dict_keyslist[n][1]).issubset(main_glac_rgi.RGIId.values.tolist())
                ds_all.loc[n, 'first_RGIId'] = ds_dict_keyslist[n][1][0]
            # Remove groups where all glaciers are not included
            ds = ds_all[ds_all.glaciers_present == True].copy()
            ds.reset_index(drop=True, inplace=True)
            # Time indices
            ds['t1_year'] = ds[self.t1_cn].astype(str).str.split('.').str[0].str[:4].astype(int)
            ds['t1_month'] = ds[self.t1_cn].astype(str).str.split('.').str[0].str[4:6].astype(int)
            ds['t1_day'] = ds[self.t1_cn].astype(str).str.split('.').str[0].str[6:].astype(int)
            ds['t2_year'] = ds[self.t2_cn].astype(str).str.split('.').str[0].str[:4].astype(int)
            ds['t2_month'] = ds[self.t2_cn].astype(str).str.split('.').str[0].str[4:6].astype(int)
            ds['t2_day'] = ds[self.t2_cn].astype(str).str.split('.').str[0].str[6:].astype(int)
            # if month/day unknown for start or end period, then replace with water year
            # Add latitude 
            latdict = dict(zip(main_glac_rgi['RGIId'], main_glac_rgi['CenLat']))
            ds['CenLat'] = ds['first_RGIId'].map(latdict)
            ds['lat_category'] = np.nan
            ds.loc[ds['CenLat'] >= input.lat_threshold, 'lat_category'] = 'northernmost'
            ds.loc[(ds['CenLat'] < input.lat_threshold) & (ds['CenLat'] > 0), 'lat_category'] = 'north'
            ds.loc[(ds['CenLat'] <= 0) & (ds['CenLat'] > -1*input.lat_threshold), 'lat_category'] = 'south'
            ds.loc[ds['CenLat'] <= -1*input.lat_threshold, 'lat_category'] = 'southernmost'
            ds['months_wintersummer'] = ds['lat_category'].map(input.monthdict)
            ds['winter_begin'] = ds['months_wintersummer'].apply(lambda x: x[0])
            ds['winter_end'] = ds['months_wintersummer'].apply(lambda x: x[1])
            ds['summer_begin'] = ds['months_wintersummer'].apply(lambda x: x[2])
            ds['summer_end'] = ds['months_wintersummer'].apply(lambda x: x[3])
            ds.loc[ds['t1_month'] == 99, 't1_month'] = ds.loc[ds['t1_month'] == 99, 'winter_begin']
            ds.loc[ds['t1_day'] == 99, 't1_day'] = 1
            ds.loc[ds['t2_month'] == 99, 't2_month'] = ds.loc[ds['t2_month'] == 99, 'winter_begin'] - 1
            for x in range(ds.shape[0]):
                if ds.loc[x, 't2_day'] == 99:
                    try:
                        ds.loc[x, 't2_day'] = (
                                dates_table.loc[(ds.loc[x, 't2_year'] == dates_table['year']) & 
                                                (ds.loc[x, 't2_month'] == dates_table['month']), 'daysinmonth']
                                                .values[0])
                    except:
                        ds.loc[x, 't2_day'] = 28    
            # Calculate decimal year and drop measurements outside of calibration period
            ds['t1_datetime'] = pd.to_datetime(
                    pd.DataFrame({'year':ds.t1_year.values, 'month':ds.t1_month.values, 'day':ds.t1_day.values}))
            ds['t2_datetime'] = pd.to_datetime(
                    pd.DataFrame({'year':ds.t2_year.values, 'month':ds.t2_month.values, 'day':ds.t2_day.values}))
            ds['t1_doy'] = ds.t1_datetime.dt.strftime("%j").astype(float)
            ds['t2_doy'] = ds.t2_datetime.dt.strftime("%j").astype(float)
            ds['t1_daysinyear'] = (
                    (pd.to_datetime(pd.DataFrame({'year':ds.t1_year.values, 'month':12, 'day':31})) - 
                     pd.to_datetime(pd.DataFrame({'year':ds.t1_year.values, 'month':1, 'day':1}))).dt.days + 1)
            ds['t2_daysinyear'] = (
                    (pd.to_datetime(pd.DataFrame({'year':ds.t2_year.values, 'month':12, 'day':31})) - 
                     pd.to_datetime(pd.DataFrame({'year':ds.t2_year.values, 'month':1, 'day':1}))).dt.days + 1)
            ds['t1'] = ds.t1_year + ds.t1_doy / ds.t1_daysinyear
            ds['t2'] = ds.t2_year + ds.t2_doy / ds.t2_daysinyear
            end_datestable = dates_table.loc[dates_table.shape[0]-1, 'date']
            end_datetime = datetime.datetime(end_datestable.year, end_datestable.month + 1, end_datestable.day)
            ds = ds[ds['t1_datetime'] >= dates_table.loc[0, 'date']]
            ds = ds[ds['t2_datetime'] < end_datetime]
            ds.reset_index(drop=True, inplace=True)
            # Time indices
            #  exclude spinup years, since massbal fxn discards spinup years
            ds['t1_idx'] = np.nan
            ds['t2_idx'] = np.nan
            for x in range(ds.shape[0]):
                ds.loc[x,'t1_idx'] = (dates_table[(ds.loc[x, 't1_year'] == dates_table['year']) & 
                                                  (ds.loc[x, 't1_month'] == dates_table['month'])].index.values[0])
                ds.loc[x,'t2_idx'] = (dates_table[(ds.loc[x, 't2_year'] == dates_table['year']) & 
                                                  (ds.loc[x, 't2_month'] == dates_table['month'])].index.values[0])
            # Mass balance [mwe]
            ds['mb_mwe'] = np.nan
            ds['mb_mwe_err'] = np.nan
            ds.loc[ds['dhdt_ma'].notnull(), 'mb_mwe'] = (
                    ds.loc[ds['dhdt_ma'].notnull(), 'dhdt_ma'] * input.density_ice / input.density_water * 
                    (ds['t2'] - ds['t1']))
            ds.loc[ds['dhdt_ma'].notnull(), 'mb_mwe_err'] = (
                    ds.loc[ds['dhdt_ma'].notnull(), 'dhdt_unc_ma'] * input.density_ice / input.density_water * 
                    (ds['t2'] - ds['t1']))
            
            # Add columns for consistency with individual glacier observations
            ds['RGIId'] = np.nan
            ds['glacno'] = np.nan
            ds['sla_m'] = np.nan
            ds['z1_idx'] = np.nan
            ds['z2_idx'] = np.nan
            ds['z1'] = np.nan
            ds['z2'] = np.nan
            ds['WGMS_ID'] = np.nan
            
        elif self.name == 'mcnabb' or self.name == 'larsen': 
            # Load all data
            ds_all = pd.read_csv(self.ds_fp + self.ds_fn)                  
            ds_all['RegO1'] = [int(x.split('-')[1].split('.')[0]) for x in ds_all[self.rgiid].values]
            # Select data for specific region
            ds_reg = ds_all[ds_all['RegO1']==self.rgi_regionO1].copy()
            ds_reg.reset_index(drop=True, inplace=True)
            # Glacier number and index for comparison
            ds_reg['glacno'] = [int(x.split('-')[1].split('.')[1]) for x in ds_all[self.rgiid].values]
            # Select glaciers with mass balance data
            ds = (ds_reg.iloc[np.where(ds_reg['glacno'].isin(main_glac_rgi['glacno']) == True)[0],:]
                  ).copy()
            ds.reset_index(drop=True, inplace=True)
            # Elevation indices
            elev_bins = main_glac_hyps.columns.values.astype(int)
            elev_bin_interval = elev_bins[1] - elev_bins[0]
            ds['z1_idx'] = (
                    (main_glac_hyps.iloc[ds['glacno'].map(glacnodict)].values != 0).argmax(axis=1).astype(int))
            ds['z2_idx'] = (
                    (main_glac_hyps.iloc[ds['glacno'].map(glacnodict)].values.cumsum(1)).argmax(axis=1).astype(int))
            # Lower and upper bin elevations [masl]
            ds['z1'] = elev_bins[ds['z1_idx'].values] - elev_bin_interval/2
            ds['z2'] = elev_bins[ds['z2_idx'].values] + elev_bin_interval/2
            # Area [km2]
            ds['area_km2'] = np.nan
            for x in range(ds.shape[0]):
                ds.loc[x,'area_km2'] = (
                        main_glac_hyps.iloc[glacnodict[ds.loc[x,'glacno']], 
                                            ds.loc[x,'z1_idx']:ds.loc[x,'z2_idx']+1].sum())
            # Time indices
#            for x in ds[self.t1_cn].values:
#                print(x)
#                print(int(x.split('-')[0]))
            try:
                ds['t1_year'] = [int(x.split('-')[0]) for x in ds[self.t1_cn].values]
                ds['t1_month'] = [int(x.split('-')[1]) for x in ds[self.t1_cn].values]
                ds['t1_day'] = [int(x.split('-')[2]) for x in ds[self.t1_cn].values]
#                ds['t1_daysinmonth'] = ds.apply(lambda row: modelsetup.daysinmonth(row['t1_year'], row['t1_month']), axis=1)
                ds['t2_year'] = [int(x.split('-')[0]) for x in ds[self.t2_cn].values]
                ds['t2_month'] = [int(x.split('-')[1]) for x in ds[self.t2_cn].values]
                ds['t2_day'] = [int(x.split('-')[2]) for x in ds[self.t2_cn].values]
            except:
                ds['t1_year'] = [int(x.split('/')[2]) for x in ds[self.t1_cn].values]
                ds['t1_month'] = [int(x.split('/')[0]) for x in ds[self.t1_cn].values]
                ds['t1_day'] = [int(x.split('/')[1]) for x in ds[self.t1_cn].values]
#                ds['t1_daysinmonth'] = ds.apply(lambda row: modelsetup.daysinmonth(row['t1_year'], row['t1_month']), axis=1)
                ds['t2_year'] = [int(x.split('/')[2]) for x in ds[self.t2_cn].values]
                ds['t2_month'] = [int(x.split('/')[0]) for x in ds[self.t2_cn].values]
                ds['t2_day'] = [int(x.split('/')[1]) for x in ds[self.t2_cn].values]
              
#            for x in range(ds.shape[0]):
#                print(ds.loc[x, self.t1_cn])
#                print(ds.loc[x, 't1_month'], ds.loc[x, 't1_year'])
#                print(modelsetup.daysinmonth(ds.loc[x, 't1_year'], ds.loc[x, 't1_month']))
#            ds['t1_year'] = [int(x.split('-')[0]) for x in ds[self.t1_cn].values]
#            ds['t1_month'] = [int(x.split('-')[1]) for x in ds[self.t1_cn].values]
#            ds['t1_day'] = [int(x.split('-')[2]) for x in ds[self.t1_cn].values]
#            ds['t1_daysinmonth'] = ds.apply(lambda row: modelsetup.daysinmonth(row['t1_year'], row['t1_month']), axis=1)
#            ds['t2_year'] = [int(x.split('-')[0]) for x in ds[self.t2_cn].values]
#            ds['t2_month'] = [int(x.split('-')[1]) for x in ds[self.t2_cn].values]
#            ds['t2_day'] = [int(x.split('-')[2]) for x in ds[self.t2_cn].values]
            ds['t1_daysinmonth'] = ds.apply(lambda row: modelsetup.daysinmonth(row['t1_year'], row['t1_month']), axis=1)
            ds['t2_daysinmonth'] = ds.apply(lambda row: modelsetup.daysinmonth(row['t2_year'], row['t2_month']), axis=1)
            ds['t1_datetime'] = pd.to_datetime(
                    pd.DataFrame({'year':ds.t1_year.values, 'month':ds.t1_month.values, 'day':ds.t1_day.values}))
            ds['t2_datetime'] = pd.to_datetime(
                    pd.DataFrame({'year':ds.t2_year.values, 'month':ds.t2_month.values, 'day':ds.t2_day.values}))
            ds['t1'] = ds['t1_year'] + (ds['t1_month'] + ds['t1_day'] / ds['t1_daysinmonth']) / 12
            ds['t2'] = ds['t2_year'] + (ds['t2_month'] + ds['t2_day'] / ds['t2_daysinmonth']) / 12
            # Remove data with dates outside of calibration period
            year_decimal_min = dates_table.loc[0,'year'] + dates_table.loc[0,'month'] / 12
            year_decimal_max = (dates_table.loc[dates_table.shape[0]-1,'year'] + 
                                (dates_table.loc[dates_table.shape[0]-1,'month'] + 1) / 12)
            ds = ds[ds['t1_year'] + ds['t1_month'] / 12 >= year_decimal_min]
            ds = ds[ds['t2_year'] + ds['t2_month'] / 12 < year_decimal_max]
            ds.reset_index(drop=True, inplace=True)    
            # Determine time indices (exclude spinup years, since massbal fxn discards spinup years)
            ds['t1_idx'] = np.nan
            ds['t2_idx'] = np.nan
            for x in range(ds.shape[0]):
                ds.loc[x,'t1_idx'] = (dates_table[(ds.loc[x, 't1_year'] == dates_table['year']) & 
                                                  (ds.loc[x, 't1_month'] == dates_table['month'])].index.values[0])
                ds.loc[x,'t2_idx'] = (dates_table[(ds.loc[x, 't2_year'] == dates_table['year']) & 
                                                  (ds.loc[x, 't2_month'] == dates_table['month'])].index.values[0])
            ds['t1_idx'] = ds['t1_idx'].astype(int)
            # Specific mass balance [mwea]
            ds['mb_mwe'] = ds[self.mb_mwea_cn] * (ds['t2'] - ds['t1'])
            ds['mb_mwe_err'] = ds[self.mb_mwea_err_cn] * (ds['t2'] - ds['t1']) 
            # Total mass change [Gt]
#            ds['mb_gt'] = ds[self.mb_vol_cn] * (ds['t2'] - ds['t1']) * (1/1000)**3 * input.density_water / 1000
#            ds['mb_gt_err'] = ds[self.mb_vol_err_cn] * (ds['t2'] - ds['t1']) * (1/1000)**3 * input.density_water / 1000
            # Observation type
            ds['obs_type'] = 'mb_geo'
            # Add columns with nan for things not in list
            ds_addcols = [x for x in ds_output_cols if x not in ds.columns.values]
            for colname in ds_addcols:
                ds[colname] = np.nan
        # Select output
        ds_output = ds[ds_output_cols].sort_values(['glacno', 't1_idx'])
        ds_output.reset_index(drop=True, inplace=True)
        return ds_output


#%% Testing
if __name__ == '__main__':
    # Glacier selection
    rgi_regionsO1 = [1]
    rgi_glac_number = 'all'
    startyear = 1980
    endyear = 2017
    
    # Select glaciers
    main_glac_rgi = modelsetup.selectglaciersrgitable(rgi_regionsO1=rgi_regionsO1, rgi_regionsO2 = 'all', 
                                                      rgi_glac_number=rgi_glac_number)
    # Glacier hypsometry [km**2], total area
    main_glac_hyps = modelsetup.import_Husstable(main_glac_rgi, rgi_regionsO1, input.hyps_filepath, 
                                                 input.hyps_filedict, input.hyps_colsdrop)
    # Determine dates_table_idx that coincides with data
    dates_table = modelsetup.datesmodelrun(startyear, endyear, spinupyears=0)
    
    elev_bins = main_glac_hyps.columns.values.astype(int)
    elev_bin_interval = elev_bins[1] - elev_bins[0]
    
    # Testing    
    mb1 = MBData(name='larsen', rgi_regionO1=rgi_regionsO1[0])
#    mb1 = MBData(name='mauer', rgi_regionO1=rgi_regionsO1[0])
#    mb1 = MBData(name='cogley', rgi_regionO1=rgi_regionsO1[0])
    ds_output = mb1.retrieve_mb(main_glac_rgi, main_glac_hyps, dates_table)
    
    #%%
#    cal_datasets = ['mcnabb', 'larsen', 'wgms_d', 'wgms_ee']
#    cal_datasets = ['wgms_d', 'wgms_ee', 'group']
    cal_datasets = ['larsen']
##    cal_datasets = ['wgms_ee']
##    cal_datasets = ['wgms_d']
##    cal_datasets = ['group']
    
    cal_data = pd.DataFrame()
    for dataset in cal_datasets:
        cal_subset = MBData(name=dataset, rgi_regionO1=rgi_regionsO1[0])
        cal_subset_data = cal_subset.retrieve_mb(main_glac_rgi, main_glac_hyps, dates_table)
        cal_data = cal_data.append(cal_subset_data, ignore_index=True)
    cal_data = cal_data.sort_values(['glacno', 't1_idx'])
    cal_data.reset_index(drop=True, inplace=True)
    
    #%%
    # Count unique glaciers and fraction of total area
    rgiid_unique = list(cal_data.RGIId.unique())
    rgiid_unique_idx = []
    for rgiid in rgiid_unique:
        rgiid_unique_idx.append(np.where(main_glac_rgi.RGIId.values == rgiid)[0][0])
    print('Glacier area covered: ', np.round(main_glac_rgi.loc[rgiid_unique_idx, 'Area'].sum() / 
                                             main_glac_rgi['Area'].sum() * 100,1),'%')

#%% PRE-PROCESS MCNABB DATA
#    # Remove glaciers that:
#    #  (1) poor percent coverage
#    #  (2) uncertainty is too high
#    # Glacier selection
#    rgi_regionsO1 = [1]
#    rgi_glac_number = 'all'
#    startyear = 1980
#    endyear = 2017
#    
#    density_ice_brun = 850
#    
#    mcnabb_fn = 'Alaska_dV_17jun.csv'
#    
#    # Select glaciers
#    main_glac_rgi = modelsetup.selectglaciersrgitable(rgi_regionsO1=rgi_regionsO1, rgi_regionsO2 = 'all', 
#                                                      rgi_glac_number=rgi_glac_number)
#    # Load data
#    ds_raw = pd.read_csv(input.mcnabb_fp + mcnabb_fn)
#    
#    # remove data with poor coverage
#    ds1 = ds_raw[ds_raw['pct_data'] > 0.75].copy()
#    ds1.reset_index(drop=True, inplace=True)
#    
#    # remove glaciers with too high uncertainty (> 1.96 stdev)
#    uncertainty_median = ds1.e_dh.median()
#    ds1['e_mad'] = np.absolute(ds1['e_dh'] - uncertainty_median)
#    uncertainty_mad = np.median(ds1['e_mad'])
#    print('uncertainty median and mad [m/yr]:', np.round(uncertainty_median,2), np.round(uncertainty_mad,2))
#    ds2 = ds1[ds1['e_dh'] < uncertainty_median + 3*uncertainty_mad].copy()
#    ds2.reset_index(drop=True, inplace=True)
#    print('Glaciers removed (too high uncertainty):', ds1.shape[0] - ds2.shape[0], 'points')
#
#    # Minimum and maximum mass balances
##    print(ds2.loc[np.where(ds2.smb.values == ds2.smb.max())[0][0],'smb'], 
##          ds2.loc[np.where(ds2.smb.values == ds2.smb.max())[0][0],'e_dh'])
##    print(ds2.loc[np.where(ds2.smb.values == ds2.smb.min())[0][0],'smb'], 
##          ds2.loc[np.where(ds2.smb.values == ds2.smb.min())[0][0],'e_dh'])
#    
#    ds2.sort_values('RGIId')
#    ds2.reset_index(drop=True, inplace=True)
#    
#    # Count unique glaciers and fraction of total area
#    rgiid_unique = list(ds2.RGIId.unique())
#    rgiid_unique_idx = []
#    for rgiid in rgiid_unique:
#        rgiid_unique_idx.append(np.where(main_glac_rgi.RGIId.values == rgiid)[0][0])
#    print('Glacier area covered: ', np.round(main_glac_rgi.loc[rgiid_unique_idx, 'Area'].sum() / 
#                                             main_glac_rgi['Area'].sum() * 100,1),'%')
#    
#    rgiid_values = list(ds2.RGIId.values)
#    rgiid_idx = []
#    for rgiid in rgiid_values:
#        rgiid_idx.append(np.where(main_glac_rgi.RGIId.values == rgiid)[0][0])
#    ds2['CenLat'] = main_glac_rgi.loc[rgiid_idx, 'CenLat'].values
#    ds2['CenLon'] = main_glac_rgi.loc[rgiid_idx, 'CenLon'].values
#    
#    ds2['mb_mwea'] = ds2['smb'] * density_ice_brun / input.density_water
#    ds2['mb_mwea_sigma'] = ds2['e_dh'] * density_ice_brun / input.density_water
#    
#    ds2.to_csv(input.mcnabb_fp + mcnabb_fn.replace('.csv','_preprocessed.csv'))
    

#%%
#    # PRE-PROCESS MAUER DATA
#    mauer_fn = 'Mauer_geoMB_HMA_1970s_2000.csv'
#    min_pctCov = 80
#    
#    ds = pd.read_csv(input.mauer_fp + mauer_fn)
#    ds.dropna(axis=0, how='any', inplace=True)
#    ds.sort_values('RGIId')
#    ds.reset_index(drop=True, inplace=True)
#    demyears = ds.demYears.tolist()
#    demyears = [x.split(';') for x in demyears]
#    t1_raw = []
#    t2 = []
#    for x in demyears:
#        if '2000' in x:
#            x.remove('2000')
#            t2.append(2000)
#            t1_raw.append([np.float(y) for y in x])
#    t1 = np.array([np.array(x).mean() for x in t1_raw])
#    ds['t1'] = t1
#    ds['t2'] = t2    
#    # Minimum percent coverage
#    ds2 = ds[ds.pctCov > min_pctCov].copy()
#    ds2['RegO1'] = ds2.RGIId.astype(int)
#    # Glacier number and index for comparison
#    ds2['glacno'] = ((ds2['RGIId'] % 1) * 10**5).round(0).astype(int)
#    ds_list = ds2[['RegO1', 'glacno']]
#    ds2['RGIId'] = ds2['RegO1'] + ds2['glacno'] / 10**5
#    ds2.reset_index(drop=True, inplace=True)
#    ds2.drop(['RegO1', 'glacno'], axis=1, inplace=True)
#    ds2.to_csv(input.mauer_fp + input.mauer_fn.split('.csv')[0] + '_min' + str(min_pctCov) + 'pctCov.csv', index=False)
#    
#    # Pickle lists of glacier numbers for each region
#    import pickle
#    for reg in [13, 14, 15]:
#        ds_subset = ds_list[ds_list['RegO1'] == reg]
#        rgi_glacno_list = [str(x).rjust(5,'0') for x in ds_subset['glacno'].tolist()]
#        pickle_fn = 'R' + str(reg) + '_mauer_1970s_2000_rgi_glac_number.pkl'
#        print('Region ' + str(reg) + ' list:', rgi_glacno_list)
#        print(pickle_fn)
##        
##        with open(pickle_fn, 'wb') as f:
##            pickle.dump(rgi_glacno_list, f)        
    
    #%%
#    import pickle
#    region = 15
#    
#    mauer_pickle_fn = 'R' + str(region) + '_mauer_1970s_2000_rgi_glac_number.pkl'
#            
#    with open(mauer_pickle_fn, 'rb') as f:
#        rgi_glac_number = pickle.load(f)
#    
#     # Select glaciers
#    main_glac_rgi = modelsetup.selectglaciersrgitable(rgi_regionsO1=[region], rgi_regionsO2 = 'all', 
#                                                      rgi_glac_number=rgi_glac_number)
#    # Glacier hypsometry [km**2], total area
#    main_glac_hyps = modelsetup.import_Husstable(main_glac_rgi, [region], input.hyps_filepath, 
#                                                 input.hyps_filedict, input.hyps_colsdrop)
#    # Determine dates_table_idx that coincides with data
#    dates_table = modelsetup.datesmodelrun(1970, 2017, spinupyears=0)
#    
#    
#    # Select mass balance data
#    mb1 = MBData(name='mauer', rgi_regionO1=region)
#    ds_mb = mb1.retrieve_mb(main_glac_rgi, main_glac_hyps, dates_table)