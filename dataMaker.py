#!/usr/bin/env python3

import csv
import os
import re
from decimal import Decimal as D, ROUND_DOWN
from datetime import datetime
import numpy as np


class DataMaker:

    __fileDir = os.path.dirname(os.path.abspath(__file__))
    __tikets = []
    # Смещение для предсказания в дня
    __pred_offset = 1
    # Число знаков в расчетных значениях
    __accuracy = '0.00001'
    # Граничные значения зон UP/DOWN
    __max_border = 0.05
    __min_border = -0.05

    __up_counter = 0
    __down_counter = 0
    __none_counter = 0


    def __init__(self, batch_size=3):

        self.__batch_size = batch_size
        # Tickers array 0 - test, 1 - long list, 2 - short list
        self.__tickers_array = [['AAL','NVS'],
                                ['A', 'AA', 'AAPL', 'AB', 'ABC', "ABCB", 'ABEO', 'ABEV', 'ABIO', 'ABM', 'ABMD', 'ABT',
                                'ACGL', 'ACHC', 'ACHV', 'ACIW', 'ACNB', 'ACU', 'ACY', 'ADBE', 'ADC', 'ADI', 'ADM',
                                'ADMP', 'ADP', 'ADSK', 'ADTN', 'ADX', 'AE', 'AEE', 'AEG', 'AEGN', 'AEHR', 'AEIS', 'AEM',
                                'AEMD', 'AEO', 'AEP', 'AES', 'AEY', 'AFG', 'AFL', 'AGCO', 'AGM-A', 'AGM', 'AGN', 'AGX',
                                'AGYS', 'AHPI', 'AI', 'AIG', 'AIN', 'AIR', 'AIRT', 'AIT', 'AIV', 'AJG', 'AJRD', 'AKAM',
                                'AKO-A', 'AKO-B', 'AKR', 'AKS', 'ALB', 'ALCO', 'ALE', 'ALG', 'ALJJ', 'ALK', 'ALKS',
                                'ALL', 'ALOT', 'ALSK', 'ALV', 'ALX', 'ALXN', 'AMAG', 'AMAT', 'AMD', 'AME', 'AMED',
                                'AMG', 'AMGN', 'AMKR', 'AMNB', 'AMOT', 'AMRB', 'AMRN', 'AMS', 'AMSC', 'AMSWA', 'AMT',
                                'AMTD', 'AMWD', 'AMZN', 'AN', 'ANAT', 'ANDE', 'ANF', 'ANH', 'ANIK', 'ANSS', 'AOBC',
                                'AON', 'AOS', 'AP', 'APA', 'APD', 'APH', 'APOG', 'APT', 'APTO', 'ARCB', 'ARCW', 'ARE',
                                'ARKR', 'ARL', 'ARLP', 'AROW', 'ARQL', 'ARTNA', 'ARTW', 'ARW', 'ARWR', 'ASA', 'ASB',
                                'ASFI', 'ASG', 'ASGN', 'ASH', 'ASML', 'ASNA', 'ASRV', 'ASRVP', 'ASTC', 'ASTE', 'ASUR',
                                'ASYS', 'ATAX', 'ATGE', 'ATI', 'ATLC', 'ATNI', 'ATO', 'ATR', 'ATRI', 'ATRO', 'ATRS',
                                'ATVI', 'AU', 'AUBN', 'AUDC', 'AUTO', 'AVA', 'AVB', 'AVD', 'AVDL', 'AVID', 'AVNW',
                                'AVP', 'AVT', 'AVX', 'AVY', 'AWF', 'AWR', 'AWRE', 'AWX', 'AXAS', 'AXDX', 'AXE', 'AXGN',
                                'AXL', 'AXP', 'AXR', 'AXTI', 'AZN', 'AZO', 'AZPN', 'AZZ', 'B', 'BA', 'BAC', 'BAM',
                                'BANF', 'BANR', 'BAP', 'BASI', 'BAX', 'BB', 'BBBY', 'BBSI', 'BBVA', 'BBY', 'BC', 'BCE',
                                'BCO', 'BCOR', 'BCPC', 'BCRX', 'BCS', 'BCV', 'BDC', 'BDGE', 'BDL', 'BDN', 'BDR', 'BDX',
                                'BEBE', 'BELFA', 'BELFB', 'BEN', 'BF-A', 'BF-B', 'BFS', 'BGCP', 'BGG', 'BHB', 'BHE',
                                'BHP', 'BIF', 'BIG', 'BIIB', 'BIO-B', 'BIO', 'BIOL', 'BJRI', 'BK', 'BKE', 'BKH', 'BKN',
                                'BKNG', 'BKSC', 'BKT', 'BKYI', 'BLDP', 'BLFS', 'BLK', 'BLL', 'BLX', 'BMI', 'BMO',
                                'BMRA', 'BMRC', 'BMRN', 'BMTC', 'BMY', 'BNS', 'BNSO', 'BOCH', 'BOKF', 'BOSC', 'BP',
                                'BPFH', 'BPOP', 'BPT', 'BRC', 'BREW', 'BRID', 'BRK-A', 'BRK-B', 'BRKL', 'BRKS', 'BRN',
                                'BRO', 'BSD', 'BSET', 'BSQR', 'BSRR', 'BSTC', 'BSX', 'BTI', 'BTO', 'BUSE', 'BVN',
                                'BVSN', 'BWA', 'BXMT', 'BXP', 'BXS', 'BYD', 'BYFC', 'C', 'CAC', 'CACC', 'CACI', 'CAE',
                                'CAG', 'CAH', 'CAJ', 'CAKE', 'CAL', 'CALM', 'CAMP', 'CAR', 'CARV', 'CASH', 'CASI',
                                'CASS', 'CASY', 'CAT', 'CATO', 'CBB', 'CBD', 'CBL', 'CBSH', 'CBT', 'CBU', 'CCBG', 'CCF',
                                'CCI', 'CCJ', 'CCK', 'CCL', 'CCNE', 'CDE', 'CDNS', 'CDOR', 'CDR', 'CDZI', 'CEA', 'CECE',
                                'CEE', 'CENT', 'CENX', 'CERN', 'CERS', 'CET', 'CETV', 'CEV', 'CFNB', 'CFR', 'CGNX',
                                'CHCO', 'CHD', 'CHDN', 'CHE', 'CHKP', 'CHL', 'CHMG', 'CHN', 'CHNR', 'CHRW', 'CHS', 'CI',
                                'CIA', 'CIB', 'CIEN', 'CIG', 'CIGI', 'CINF', 'CIR', 'CIVB', 'CIX', 'CKH', 'CKX', 'CL',
                                'CLB', 'CLBS', 'CLCT', 'CLF', 'CLI', 'CLWT', 'CLX', 'CM', 'CMCL', 'CMCO', 'CMCSA',
                                'CMCT', 'CMD', 'CMI', 'CMO', 'CMS', 'CMT', 'CMTL', 'CNBKA', 'CNI', 'CNMD', 'CNOB',
                                'CNP', 'CNTY', 'COF', 'COG', 'COHR', 'COHU', 'COKE', 'COLB', 'COLM', 'COO', 'COP',
                                'COST', 'COT', 'CP', 'CPB', 'CPE', 'CPK', 'CPRT', 'CPSH', 'CPSS', 'CPT', 'CR', 'CRD-B',
                                'CREE', 'CRESY', 'CRF', 'CRH', 'CRK', 'CRMT', 'CRS', 'CRVL', 'CRY', 'CS', 'CSCO',
                                'CSGP', 'CSGS', 'CSL', 'CSPI', 'CSS', 'CSV', 'CSWC', 'CSX', 'CTAS', 'CTB', 'CTHR',
                                'CTIC', 'CTL', 'CTSH', 'CTXS', 'CUB', 'CUBA', 'CULP', 'CUZ', 'CVBF', 'CVLY', 'CVM',
                                'CVTI', 'CVX', 'CW', 'CWBC', 'CWCO', 'CWST', 'CWT', 'CX', 'CXH', 'CXW', 'CYAN', 'CYBE',
                                'CYD', 'CYH', 'CYRN', 'CYTR', 'CZNC', 'D', 'DAKT', 'DB', 'DCI', 'DCO', 'DD', 'DDF',
                                'DDR', 'DDS', 'DDT', 'DE', 'DECK', 'DENN', 'DEO', 'DGICB', 'DGII', 'DHI', 'DHR', 'DIN',
                                'DIS', 'DISH', 'DIT', 'DJCO', 'DLHC', 'DLTR', 'DMF', 'DNI', 'DNR', 'DO', 'DOV', 'DPW',
                                'DRE', 'DRI', 'DSGX', 'DSPG', 'DSS', 'DSU', 'DSWL', 'DTE', 'DUC', 'DUK', 'DVA', 'DVCR',
                                'DVD', 'DVN', 'DWSN', 'DX', 'DXC', 'DXLG', 'DXPE', 'DXR', 'DY', 'EA', 'EBAY', 'EBF',
                                'EBIX', 'ECF', 'ECL', 'ECPG', 'ED', 'EDUC', 'EE', 'EEA', 'EEFT', 'EFOI', 'EFX', 'EGBN',
                                'EGOV', 'EGY', 'EIX', 'EL', 'ELGX', 'ELLO', 'ELP', 'ELTK', 'ELY', 'EME', 'EMF', 'EMITF',
                                'EMKR', 'EML', 'EMMS', 'EMN', 'EMR', 'ENB', 'ENSV', 'EPAY', 'EPD', 'EPM', 'EPR', 'EQC',
                                'EQR', 'EQS', 'EQT', 'ERIC', 'ES', 'ESBK', 'ESCA', 'ESE', 'ESGR', 'ESP', 'ESS', 'ESTE',
                                'ETFC', 'ETH', 'ETM', 'ETN', 'ETR', 'EV', 'EVF', 'EVI', 'EVN', 'EVOL', 'EWBC', 'EXC',
                                'EXP', 'EXPO', 'EXTR', 'EZPW', 'F', 'FARM', 'FARO', 'FAST', 'FAX', 'FBNC', 'FBSS', 'FC',
                                'FCAP', 'FCBC', 'FCCY', 'FCEL', 'FCFS', 'FCN', 'FCNCA', 'FCO', 'FDBC', 'FDS', 'FDX',
                                'FEIM', 'FELE', 'FFBC', 'FFIC', 'FFIN', 'FFIV', 'FHN', 'FICO', 'FII', 'FISV', 'FITB',
                                'FIX', 'FIZZ', 'FL', 'FLEX', 'FLIC', 'FLL', 'FLO', 'FLS', 'FLWS', 'FMBI', 'FMC', 'FMNB',
                                'FMS', 'FMX', 'FNB', 'FNHC', 'FNLC', 'FOE', 'FONR', 'FORD', 'FORR', 'FORTY', 'FOSL',
                                'FRD', 'FRME', 'FRPH', 'FRT', 'FSI', 'FSS', 'FSTR', 'FTEK', 'FTR', 'FUL', 'FULT',
                                'FUNC', 'FUND', 'FWRD', 'GAB', 'GAIA', 'GAM', 'GATX', 'GBCI', 'GBL', 'GBR', 'GBX',
                                'GCBC', 'GCI', 'GCO', 'GCV', 'GDEN', 'GE', 'GEC', 'GEF', 'GEL', 'GEOS', 'GERN', 'GES',
                                'GF', 'GFF', 'GFI', 'GGB', 'GGG', 'GHC', 'GIII', 'GIL', 'GILD', 'GILT', 'GIM', 'GIS',
                                'GLBZ', 'GLW', 'GNTX', 'GOGL', 'GOLD', 'GPC', 'GPI', 'GPOR', 'GPX', 'GRA', 'GRC',
                                'GROW', 'GS', 'GSBC', 'GSK', 'GT', 'GTIM', 'GTN-A', 'GUT', 'GV', 'GVA', 'GWW', 'HA',
                                'HAE', 'HAIN', 'HAL', 'HALL', 'HBAN', 'HCKT', 'HE', 'HEI-A', 'HELE', 'HES', 'HFC',
                                'HHS', 'HIBB', 'HIG', 'HIHO', 'HIX', 'HL', 'HLX', 'HMC', 'HMG', 'HMNF', 'HMNY', 'HMSY',
                                'HMY', 'HNI', 'HNP', 'HNRG', 'HOG', 'HOLX', 'HOPE', 'HOV', 'HP', 'HPQ', 'HQH', 'HQL',
                                'HR', 'HRB', 'HRC', 'HRL', 'HSBC', 'HSC', 'HSIC', 'HSII', 'HSKA', 'HST', 'HSY', 'HTLF',
                                'HUBB', 'HUBG', 'HUM', 'HURC', 'HVT-A', 'HVT', 'HWBK', 'HWKN', 'HZO', 'IAC', 'IAF',
                                'IART', 'IBKC', 'IBOC', 'ICAD', 'ICCC', 'ICLR', 'IDA', 'IDCC', 'IDN', 'IDRA', 'IDXG',
                                'IDXX', 'IEC', 'IEP', 'IESC', 'IEX', 'IFN', 'IHC', 'IHT', 'IIF', 'IIIN', 'IIM', 'IIN',
                                'IIVI', 'IKNX', 'IMAX', 'IMGN', 'IMH', 'IMMR', 'IMMU', 'IMO', 'INAP', 'INCY', 'INDB',
                                'INFY', 'ING', 'INGR', 'INO', 'INOD', 'INS', 'INSI', 'INTC', 'INTG', 'INTL', 'INTT',
                                'INTU', 'INVE', 'IO', 'IOR', 'IOSP', 'IP', 'IPAR', 'IPG', 'IQI', 'IR', 'IRIX', 'IRM',
                                'IRS', 'ISIG', 'ISNS', 'IT', 'ITIC', 'ITT', 'ITW', 'IVAC', 'IVC', 'IX', 'JAKK', 'JBL',
                                'JBSS', 'JCI', 'JCOM', 'JCP', 'JCTCF', 'JEQ', 'JHI', 'JHS', 'JHX', 'JJSF', 'JMM', 'JNJ',
                                'JNPR', 'JOB', 'JOF', 'JOUT', 'JPM', 'JW-A', 'JW-B', 'JWN', 'KAI', 'KAMN', 'KBAL',
                                'KELYA', 'KELYB', 'KEM', 'KEP', 'KEQU', 'KEX', 'KEY', 'KGC', 'KIM', 'KINS', 'KLAC',
                                'KLIC', 'KMB', 'KMT', 'KMX', 'KNX', 'KO', 'KOF', 'KOPN', 'KOSS', 'KRC', 'KSS', 'KTF',
                                'KTP', 'KWR', 'L', 'LAKE', 'LAMR', 'LANC', 'LARK', 'LAWS', 'LB', 'LBAI', 'LBY', 'LCII',
                                'LCNB', 'LCUT', 'LDL', 'LECO', 'LEE', 'LEG', 'LEN', 'LEO', 'LFUS', 'LGF-A', 'LGND',
                                'LH', 'LKFN', 'LLY', 'LMT', 'LNC', 'LNG', 'LNT', 'LOAN', 'LOGI', 'LOW', 'LPTH', 'LPX',
                                'LRCX', 'LSCC', 'LSI', 'LTC', 'LTM', 'LUV', 'LWAY', 'LXP', 'LXU', 'LYTS', 'LZB', 'M',
                                'MAGS', 'MAN', 'MANH', 'MAR', 'MAS', 'MAT', 'MATX', 'MAYS', 'MBI', 'MBOT', 'MBWM',
                                'MCA', 'MCD', 'MCF', 'MCHP', 'MCI', 'MCK', 'MCO', 'MCR', 'MCS', 'MCY', 'MD', 'MDC',
                                'MDP', 'MDRX', 'MDT', 'MDU', 'MED', 'MEI', 'MEOH', 'MERC', 'MFA', 'MFC', 'MFIN', 'MFL',
                                'MFM', 'MFSF', 'MFT', 'MFV', 'MGEE', 'MGF', 'MGM', 'MGPI', 'MHD', 'MHF', 'MHO', 'MICR',
                                'MIN', 'MIND', 'MINI', 'MITK', 'MKC', 'MKSI', 'MLHR', 'MLI', 'MLM', 'MLR', 'MLSS',
                                'MMAC', 'MMM', 'MMS', 'MMSI', 'MMT', 'MMU', 'MNP', 'MNR', 'MNRO', 'MNST', 'MO', 'MOD',
                                'MOG-A', 'MOG-B', 'MOS', 'MOV', 'MPA', 'MPAA', 'MPB', 'MPVD', 'MQT', 'MQY', 'MRCY',
                                'MRK', 'MRO', 'MSB', 'MSD', 'MSEX', 'MSFT', 'MSI', 'MSM', 'MSN', 'MSTR', 'MT', 'MTB',
                                'MTD', 'MTEX', 'MTG', 'MTH', 'MTN', 'MTR', 'MTRN', 'MTRX', 'MTSC', 'MTX', 'MTZ', 'MU',
                                'MUA', 'MUE', 'MUH', 'MUJ', 'MUS', 'MUX', 'MVF', 'MVT', 'MXC', 'MXF', 'MXIM', 'MYE',
                                'MYGN', 'MYI', 'MYJ', 'MYL', 'MYN', 'MZA', 'NAC', 'NAD', 'NAN', 'NAT', 'NATH', 'NAV',
                                'NAVB', 'NAZ', 'NBIX', 'NBN', 'NC', 'NCA', 'NCR', 'NDSN', 'NE', 'NEN', 'NEON', 'NEU',
                                'NHC', 'NHLD', 'NHTC', 'NI', 'NICE', 'NIM', 'NJR', 'NKE', 'NKSH', 'NKTR', 'NL', 'NLY',
                                'NMI', 'NMR', 'NMY', 'NNBR', 'NNN', 'NOM', 'NOVT', 'NPK', 'NRIM', 'NRT', 'NSC', 'NSEC',
                                'NSP', 'NSSC', 'NSYS', 'NTAP', 'NTCT', 'NTIC', 'NTN', 'NTP', 'NTRS', 'NTWK', 'NTZ',
                                'NUE', 'NUM', 'NUS', 'NUV', 'NVAX', 'NVDA', 'NVEC', 'NVO', 'NVS', 'NWBI', 'NWL', 'NWLI',
                                'NWPX', 'NXC', 'NXN', 'NXQ', 'NYMX', 'O', 'OCFC', 'OCN', 'ODC', 'ODP', 'OFC', 'OFG',
                                'OFIX', 'OGE', 'OHI', 'OI', 'OII', 'OKE', 'OLED', 'OLN', 'OLP', 'OMEX', 'OMI', 'OMN',
                                'ONB', 'OPK', 'OPOF', 'OPY', 'ORAN', 'ORCL', 'ORLY', 'OSIS', 'OSUR', 'OTEX', 'OVBC',
                                'OXY', 'PAAS', 'PAR', 'PATK', 'PAYX', 'PBCT', 'PBIO', 'PBT', 'PCAR', 'PCF', 'PCG',
                                'PCH', 'PCM', 'PCTI', 'PCYG', 'PCYO', 'PDCO', 'PDEX', 'PDLI', 'PDS', 'PDT', 'PEBK',
                                'PEBO', 'PEG', 'PEGA', 'PEI', 'PENN', 'PEO', 'PEP', 'PESI', 'PFBC', 'PFBI', 'PFBX',
                                'PFE', 'PFH', 'PFIN', 'PGC', 'PGR', 'PH', 'PHI', 'PHM', 'PHX', 'PICO', 'PIR', 'PKE',
                                'PKOH', 'PKX', 'PLAB', 'PLCE', 'PLD', 'PLPC', 'PLT', 'PLUG', 'PLUS', 'PLX', 'PLXS',
                                'PMD', 'PMM', 'PMO', 'PNM', 'PNR', 'PNW', 'POL', 'POOL', 'POPE', 'POWI', 'POWL', 'PPC',
                                'PPIH', 'PPR', 'PPT', 'PRA', 'PRCP', 'PRFT', 'PRGS', 'PRGX', 'PRK', 'PRKR', 'PROV',
                                'PRPH', 'PSA', 'PSB', 'PSMT', 'PSO', 'PTNR', 'PVH', 'PW', 'PWR', 'PXD', 'PZZA', 'QCOM',
                                'QCRH', 'QDEL', 'QGEN', 'QUIK', 'QUMU', 'R', 'RAD', 'RADA', 'RAND', 'RBA', 'RBCAA',
                                'RCG', 'RCI', 'RCII', 'RCKY', 'RCL', 'RCMT', 'RCS', 'RDCM', 'RDI', 'RDN', 'RDNT',
                                'RDS-B', 'RDWR', 'RE', 'REG', 'REGN', 'RELV', 'REV', 'REX', 'RF', 'RFI', 'RFIL', 'RGCO',
                                'RGEN', 'RGLD', 'RGR', 'RGS', 'RHI', 'RHP', 'RICK', 'RIG', 'RIO', 'RJF', 'RL', 'RLH',
                                'RLI', 'RMBS', 'RMCF', 'RMD', 'RMT', 'RMTI', 'RNWK', 'ROG', 'ROK', 'ROL', 'ROP', 'ROST',
                                'RPM', 'RPT', 'RRC', 'RS', 'RTN', 'RVLT', 'RVT', 'RY', 'RYN', 'S', 'SAH', 'SAL', 'SALM',
                                'SANM', 'SASR', 'SBBX', 'SBCF', 'SBFG', 'SBGI', 'SBI', 'SBSI', 'SBUX', 'SCCO', 'SCHN',
                                'SCHW', 'SCI', 'SCKT', 'SCL', 'SCON', 'SCS', 'SCSC', 'SCVL', 'SCX', 'SEAC', 'SEB',
                                'SEIC', 'SELF', 'SENEA', 'SENEB', 'SF', 'SFNC', 'SGA', 'SGC', 'SGMA', 'SGRP', 'SGU',
                                'SHEN', 'SHLO', 'SHOO', 'SHW', 'SID', 'SIEB', 'SIF', 'SIFY', 'SIG', 'SIGI', 'SIGM',
                                'SILC', 'SIM', 'SIRI', 'SJI', 'SJM', 'SJT', 'SKM', 'SKY', 'SLB', 'SLGN', 'SLM', 'SLP',
                                'SM', 'SMBC', 'SMED', 'SMG', 'SMIT', 'SMP', 'SMSI', 'SMTC', 'SNA', 'SNBR', 'SNE',
                                'SNFCA', 'SNN', 'SNPS', 'SNV', 'SO', 'SOR', 'SPAR', 'SPB', 'SPGI', 'SPH', 'SPNS',
                                'SPPI', 'SPXC', 'SQM', 'SR', 'SRCL', 'SRDX', 'SRE', 'SRI', 'SRPT', 'SRT', 'SSB', 'SSD',
                                'SSL', 'SSP', 'SSRM', 'SSY', 'SSYS', 'STAA', 'STAR', 'STBA', 'STC', 'STE', 'STKL',
                                'STLY', 'STM', 'STRA', 'STRL', 'STRM', 'STRS', 'STT', 'STZ', 'SU', 'SUI', 'SUN', 'SUP',
                                'SVT', 'SWK', 'SWKS', 'SWM', 'SWN', 'SWX', 'SWZ', 'SYK', 'SYKE', 'SYNL', 'SYPR', 'SYX',
                                'SYY', 'T', 'TAIT', 'TARO', 'TATT', 'TAYD', 'TCBK', 'TCCO', 'TCF', 'TCI', 'TCO', 'TCX',
                                'TD', 'TDF', 'TDW', 'TECD', 'TECH', 'TELL', 'TEN', 'TENX', 'TER', 'TESS', 'TEUM',
                                'TEVA', 'TFX', 'TGA', 'TGB', 'TGC', 'TGI', 'TGNA', 'TGS', 'TGT', 'THC', 'THG', 'THO',
                                'THRM', 'TIF', 'TISI', 'TIVO', 'TJX', 'TK', 'TKR', 'TLF', 'TLGT', 'TLI', 'TLK', 'TLRD',
                                'TM', 'TMP', 'TPC', 'TPL', 'TR', 'TRC', 'TREX', 'TRIB', 'TRMB', 'TRMK', 'TRN', 'TRNS',
                                'TROW', 'TRP', 'TRST', 'TRT', 'TRV', 'TRXC', 'TSBK', 'TSCO', 'TSEM', 'TSI', 'TSM',
                                'TSN', 'TSU', 'TTC', 'TTEC', 'TTEK', 'TTI', 'TTWO', 'TU', 'TUES', 'TUP', 'TURN', 'TV',
                                'TVC', 'TVE', 'TVTY', 'TWI', 'TWMC', 'TWN', 'TXN', 'TY', 'TYL', 'UBA', 'UBCP', 'UBSI',
                                'UCFC', 'UEIC', 'UFCS', 'UFI', 'UFPI', 'UFPT', 'UG', 'UGI', 'UGP', 'UHAL', 'UHS', 'UHT',
                                'UL', 'UMH', 'UMPQ', 'UN', 'UNB', 'UNM', 'UNP', 'UNT', 'UNTY', 'UPS', 'URBN', 'URI',
                                'USA', 'USAK', 'USAP', 'USAT', 'USATP', 'USB', 'USEG', 'USLM', 'USM', 'USNA', 'UTHR',
                                'UTL', 'UTMD', 'UUU', 'UVSP', 'UVV', 'VALU', 'VAR', 'VBF', 'VBFC', 'VCEL', 'VCF',
                                'VECO', 'VEON', 'VERU', 'VFC', 'VFL', 'VGM', 'VGZ', 'VHC', 'VHI', 'VIAV', 'VICR', 'VIV',
                                'VIVO', 'VKI', 'VKQ', 'VLGEA', 'VLO', 'VLY', 'VMC', 'VMI', 'VMO', 'VNO', 'VOXX', 'VRSN',
                                'VRTX', 'VSEC', 'VSH', 'VTN', 'VTNR', 'VTR', 'WAB', 'WAFD', 'WASH', 'WAT', 'WBA', 'WBK',
                                'WBS', 'WCC', 'WCFB', 'WCN', 'WDC', 'WDFC', 'WDR', 'WEN', 'WERN', 'WETF', 'WFC', 'WGO',
                                'WHLM', 'WHR', 'WINA', 'WLFC', 'WMT', 'WNC', 'WOR', 'WPC', 'WRE', 'WRI', 'WRLD', 'WSBC',
                                'WSM', 'WSO-B', 'WSO', 'WST', 'WTBA', 'WTFC', 'WTM', 'WTR', 'WTS', 'WVFC', 'WVVI',
                                'WWD', 'WWE', 'WWR', 'WWW', 'WY', 'WYY', 'X', 'XEL', 'XLNX', 'XOM', 'XOMA', 'XRAY', 'Y',
                                'YORW', 'YRCW', 'YUM', 'ZBRA', 'ZION', 'ZIXI', 'ZNH', 'ZTR'],

                                ['MSFT', 'AAPL', 'AMZN', 'FB', 'GOOGL', 'TSLA',
                                'INTC', 'NVDA', 'NFLX', 'ADBE', 'PYPL', 'CSCO', 'PEP', 'KBAL', 'RF',
                                'CMCSA', 'AMGN', 'COST', 'TMUS', 'AVGO', 'TXN', 'VEON', 'UGI', 'RIG',
                                'QCOM', 'SBUX', 'INTU', 'MDLZ', 'BKNG', 'NVS',
                                'ISRG', 'FISV', 'REGN', 'ADP', 'ATVI', 'JD', 'AMAT', 'BBVA', 'CAR',
                                'ILMN', 'MU', 'CSX', 'ADSK', 'MELI', 'LRCX', 'ADI', 'DOV', 'WRI',
                                'EBAY', 'KHC', 'EA', 'LULU', 'WBA', 'FNB',
                                'EXC', 'BIDU', 'WDAY', 'NXPI', 'VFC', 'FMC', 'UFPI',
                                'KLAC', 'ORLY', 'SPLK', 'ROST', 'CTSH', 'SNPS', 'HSIC',
                                'ASML', 'IDXX', 'MAR', 'CSGP', 'CTAS', 'VRSK', 'CDNS',
                                'PAYX', 'PCAR', 'MCHP', 'ANSS', 'SIRI', 'FAST', 'ALXN', 'CBSH',
                                'VRSN', 'XLNX', 'INCY', 'SWKS', 'ALGN', 'DLTR', 'RYN', 'MDC',
                                'CPRT', 'CTXS', 'CHKP', 'MXIM', 'CDW', 'TCOM', 'CERN',
                                'WDC', 'EXPE', 'ULTA', 'NTAP', 'LBTYK', 'ASH', 'EQT', 'TSM', 'APD',
                                'LBTYA', 'HPQ', 'DXC', 'CAG', 'MAS', 'SPXC', 'TGNA', 'VAR']]

    # Вычисление изменения текущего значения относительно базового
    def __change_percent(self, base, curr):

        try:
            return float(D((float(curr) - float(base))/float(base)).quantize(D(self.__accuracy), rounding=ROUND_DOWN))
        except ZeroDivisionError:
            return float(1)

    # Запись данных в СSV файл
    def __append_to_file(self, ticker, data, outputDir):

        filename = outputDir + 'train_' + ticker + '.csv'
        if os.path.isfile(filename):
            os.remove(filename)

        with open(filename, 'a', newline = '') as csv_out_file:
            output = csv.writer(csv_out_file, delimiter=',')

            output.writerow(['Date', 'High', 'Low', 'Open', 'Close', 'Volume', 'Adj Close',
                            'DAY', 'C0', "Low0", "High0", "Vol", 'C1', "Low1", "High1", "Current1",
                             "H1", "L1", "F1", "Y1"])

            for line in data:
                output.writerow(line)

        csv_out_file.close()

    # Расчет данных
    def prepare_data(self, type, list_num = 1):
        """
        :param type: Тип данных (edu or test)
        :param list_num: Номер списка из массива __tickers_array
        """
        inputDir = self.__fileDir + '/data'
        outputDir = inputDir

        if (type == 'edu'):

            inputDir = inputDir + '/stocks/'
            outputDir = outputDir + '/rawdata/'
            print(" --- Prepare EDU data from " + 'Long tickers list' if (
                        list_num == 1) else 'Short tickers list' + ' ----')

        elif(type == 'test'):

            inputDir = inputDir + '/test/'
            outputDir = outputDir + '/test/rawdata/'
            print(" --- Prepare TEST data from " + 'Long tickers list' if (
                        list_num == 1) else 'Short tickers list' + ' ----')

        self.__prepare_data(list_num, inputDir, outputDir)

    # Создание массивов
    def get_Xy_arrays(self, type, list_num, prefix):

        inputDir = self.__fileDir + '/data'
        outputDir = inputDir

        if (type == 'edu'):

            inputDir = inputDir + '/rawdata/'
            outputDir = outputDir + '/'
            print(" --- Prepare EDU data from " + 'Long tickers list' if (
                    list_num == 1) else 'Short tickers list' + ' ----')

        elif (type == 'test'):

            inputDir = inputDir + '/test/rawdata/'
            outputDir = outputDir + '/test/cases/binary/'
            print(" --- Prepare TEST data from " + 'Long tickers list' if (
                    list_num == 1) else 'Short tickers list' + ' ----')

        tickers = self.__tickers_array[list_num]

        y_array = []
        X_array = []

        for __ticker in tickers:

            y_array_ticker = []
            X_array_ticker = []

            filename = inputDir + 'train_' + __ticker + '.csv'
            raw_data = []
            print("->> " + __ticker)
            with open(filename, newline='') as f:
                next(f)
                rows = csv.reader(f, delimiter=',', quotechar='|')
                i = 0
                # Выборка данных под Х и y массивы
                for row in rows:
                    i +=1
                    if (i > self.__batch_size):
                        y = list(map(int, re.findall(r'[(\d)]', list(row).pop(-1))))
                        y_array_ticker.append(y)

                        self.__up_down_none_count(y, i)

                    raw_data.append(row[8:16])
            print ("Tickers : "+str(len(y_array_ticker)))
            print("UP : "+str(self.__up_counter)+", DOWN: "+ str(self.__down_counter)+", NONE: "+str(self.__none_counter))



    # Подсчет количества UP/DOWN/NONE
    def __up_down_none_count(self, y, i):

        try:
            if (y[1] == 1):
                self.__none_counter += 1
            elif (y[0] == 1):
                self.__up_counter += 1
            elif (y[2] == 1):
                self.__down_counter += 1
        except IndexError:
            print(" ---- Error str : "+ str(i))
            print(y)
            input("Press any key .....")



    # Расчет относительных данных
    def __prepare_data(self, list_num, inputDir, outputDir):

        tickers = self.__tickers_array[list_num]
        error_tickers = []

        for __ticker in tickers:

            raw_data = []

            try:
                with open(inputDir + 'train_' + __ticker + '.csv', newline='') as f:

                    rows = csv.DictReader(f, delimiter=',', quotechar='|')
                    row = next(rows)

                    while True:

                        try:
                            next_row = next(rows)

                            # Find carrier as a change of open in percentages
                            c0 = self.__change_percent(row['Close'], next_row['Open'])
                            c1 = self.__change_percent(row['Open'], next_row['Open'])
                            volume = self.__change_percent(row['Volume'], next_row['Volume'])
                            low_0 = self.__change_percent(row['Close'], next_row['Low'])
                            high_0 = self.__change_percent(row['Close'], next_row['High'])

                            row = next_row

                            # Find day of year
                            day_of_year = self.__day_of_year(row['Date'])

                            low_1 = self.__change_percent(row['Open'], row['Low'])
                            high_1 = self.__change_percent(row['Open'], row['High'])
                            close_current_1 = self.__change_percent(row['Open'], row['Close'])

                            __row = list(row.values())
                            __row.append(day_of_year)

                            __row.append(c0)
                            __row.append(low_0)
                            __row.append(high_0)
                            __row.append(volume)

                            __row.append(c1)
                            __row.append(low_1)
                            __row.append(high_1)
                            __row.append(close_current_1)

                            raw_data.append(__row)

                        except StopIteration:
                            break

                f.close()
                print("---- " + __ticker + " ----------")
                self.__append_to_file(__ticker, self.__calculate_Y_values(raw_data, self.__pred_offset), outputDir)

            except FileNotFoundError:
                print(inputDir + 'train_' + __ticker + '.csv')
                error_tickers.append(__ticker)
                continue
        if (len(error_tickers) > 0):
            print ("Tickers not found : "+ str(error_tickers))


    # Convert date to day of year
    def __day_of_year(self, date_str):
        return datetime.strptime(date_str, '%Y-%m-%d').date().timetuple().tm_yday

    # Расчет значений Y
    def __calculate_Y_values(self, raw_data, offset):

        for i in range(len(raw_data)):
            if (i < self.__batch_size):
                continue
            else:
                __base = raw_data[i-1][4]
                __offset_array = raw_data[i:i+offset]
                __max = self.__find_ext('max', __offset_array)
                __min = self.__find_ext('min', __offset_array)
                __vector = self.__find_trend_vector(self.__change_percent(__base, __max),
                                                    self.__change_percent(__base,__min))

                raw_data[i].append(__max)
                raw_data[i].append(__min)
                raw_data[i].append(__vector)
                raw_data[i].append(self.__make_binary_y_array(__vector))

        return raw_data

    # Формирование бинарного массива y
    def __make_binary_y_array(self, __vector):

        if (__vector >= self.__max_border):
            return np.array([1,0,0])

        elif ((__vector > self.__min_border) and (__vector < self.__max_border)):
            return np.array([0, 1, 0])

        elif (__vector <= self.__min_border):
            return np.array([0, 0, 1])

    # Определение вектора тренда
    def __find_trend_vector(self, max, min):

        if (max > 0 and min >= 0):
            return max

        elif (max >= 0 and min < 0):
            return max + min

        else:
            return min

    # Поиск максимума или миниммума за период
    def __find_ext(self, type, data):

        n_array = (np.delete(np.array(data), [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], axis=1)).astype(np.float64)

        return np.amax(n_array) if (type == 'max') else np.amin(n_array)

    #
    # Save arrat to file
    def __save_numpy_array(self, outputDir, name, data):

        filename = outputDir + name + '.npy'
        if os.path.isfile(filename):
            os.remove(filename)

        with open(filename, 'wb') as f:
            np.save(f, data)
        f.close()




