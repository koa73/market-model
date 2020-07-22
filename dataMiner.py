import csv
import os
import re
from decimal import Decimal as D, ROUND_DOWN
from datetime import datetime
import numpy as np
import datetime,time


class DataMiner:

    __fileDir = os.path.dirname(os.path.abspath(__file__))
    __tikets =[]

    def __init__(self, batch_size):

        self.__batch_size = batch_size
        self.__accuracy =  '0.01'
        #self.__list_tickers()
        #self.__read_data_new()
        #self.check_dictionary('/data/output/')

        self.__tickers_array = ['A', 'AA', 'AABA', 'AAL', 'AAMC', 'AAME', 'AAN', 'AAOI', 'AAON', 'AAP', 'AAPL', 'AAT', 'AAU',
                           'AAWW', 'AAXN', 'AB', 'ABAX', 'ABB', 'ABBV', 'ABC', 'ABCB', 'ABCD', 'ABCO', 'ABDC', 'ABE',
                           'ABEO', 'ABEV', 'ABG', 'ABIO', 'ABM', 'ABMD', 'ABR', 'ABT', 'ABTX', 'ABUS', 'AC', 'ACAD',
                           'ACBI', 'ACC', 'ACCO', 'ACER', 'ACFC', 'ACGL', 'ACH', 'ACHC', 'ACHV', 'ACIW', 'ACLS', 'ACM',
                           'ACN', 'ACNB', 'ACOR', 'ACP', 'ACRE', 'ACRS', 'ACRX', 'ACST', 'ACTA', 'ACTG', 'ACU', 'ACV',
                           'ACY', 'ADAP', 'ADBE', 'ADC', 'ADES', 'ADI', 'ADM', 'ADMA', 'ADMP', 'ADMS', 'ADP', 'ADRO',
                           'ADS', 'ADSK', 'ADTN', 'ADUS', 'ADVM', 'ADX', 'ADXS', 'AE', 'AEB', 'AED', 'AEE', 'AEG',
                           'AEGN', 'AEH', 'AEHR', 'AEIS', 'AEL', 'AEM', 'AEMD', 'AEO', 'AEP', 'AER', 'AERI', 'AES',
                           'AET', 'AEY', 'AEZS', 'AFAM', 'AFB', 'AFC', 'AFG', 'AFGH', 'AFH', 'AFI', 'AFL', 'AFMD',
                           'AFSI', 'AFT', 'AFTY', 'AG', 'AGCO', 'AGD', 'AGEN', 'AGFS', 'AGGP', 'AGGY', 'AGI', 'AGIO',
                           'AGM-A', 'AGM', 'AGN', 'AGNC', 'AGO', 'AGR', 'AGRO', 'AGRX', 'AGTC', 'AGX', 'AGYS', 'AHC',
                           'AHGP', 'AHH', 'AHL', 'AHPI', 'AHT', 'AI', 'AIC', 'AIF', 'AIG', 'AIMC', 'AIMT', 'AIN',
                           'AINC', 'AINV', 'AIR', 'AIRI', 'AIRT', 'AIT', 'AIV', 'AIW', 'AIZ', 'AJG', 'AJRD', 'AJX',
                           'AKAM', 'AKBA', 'AKER', 'AKG', 'AKO-A', 'AKO-B', 'AKR', 'AKRX', 'AKS', 'AKTS', 'AKTX', 'AL',
                           'ALB', 'ALBO', 'ALCO', 'ALDX', 'ALE', 'ALEX', 'ALG', 'ALGN', 'ALGT', 'ALJJ', 'ALK', 'ALKS',
                           'ALL', 'ALLE', 'ALLT', 'ALLY', 'ALNY', 'ALO', 'ALOG', 'ALOT', 'ALPN', 'ALRM', 'ALSK', 'ALSN',
                           'ALTY', 'ALV', 'ALX', 'ALXN', 'AMAG', 'AMAT', 'AMBA', 'AMBC', 'AMC', 'AMCX', 'AMD', 'AME',
                           'AMED', 'AMG', 'AMGN', 'AMH', 'AMKR', 'AMN', 'AMNB', 'AMOT', 'AMOV', 'AMP', 'AMPE', 'AMPH',
                           'AMRB', 'AMRC', 'AMRK', 'AMRN', 'AMRS', 'AMS', 'AMSC', 'AMSF', 'AMSWA', 'AMT', 'AMTD',
                           'AMTX', 'AMWD', 'AMX', 'AMZA', 'AMZN', 'AN', 'ANAT', 'ANCB', 'ANCX', 'ANDE', 'ANDV', 'ANET',
                           'ANF', 'ANGI', 'ANGO', 'ANH', 'ANIK', 'ANIP', 'ANSS', 'ANTH', 'ANTM', 'ANY', 'AOBC', 'AOD',
                           'AON', 'AOS', 'AOSL', 'AOXG', 'AP', 'APA', 'APAM', 'APD', 'APDN', 'APEI', 'APEN', 'APH',
                           'APLE', 'APLP', 'APO', 'APOG', 'APPF', 'APPS', 'APT', 'APTO', 'APTS', 'APWC', 'AQMS', 'AQN',
                           'AR', 'ARAY', 'ARC', 'ARCB', 'ARCC', 'ARCH', 'ARCO', 'ARCW', 'ARDC', 'ARDX', 'ARE', 'ARES',
                           'ARI', 'ARII', 'ARKK', 'ARKR', 'ARKW', 'ARL', 'ARLP', 'ARMK', 'ARNA', 'AROC', 'AROW', 'ARQL',
                           'ARR', 'ARRS', 'ARTNA', 'ARTW', 'ARTX', 'ARW', 'ARWR', 'ASA', 'ASB', 'ASC', 'ASET', 'ASFI',
                           'ASG', 'ASGN', 'ASH', 'ASHX', 'ASM', 'ASMB', 'ASML', 'ASNA', 'ASND', 'ASPN', 'ASPS', 'ASPU',
                           'ASR', 'ASRV', 'ASRVP', 'ASTC', 'ASTE', 'ASUR', 'ASX', 'ASYS', 'AT', 'ATAX', 'ATEC', 'ATEN',
                           'ATGE', 'ATHM', 'ATHX', 'ATI', 'ATLC', 'ATLO', 'ATNI', 'ATNM', 'ATO', 'ATOS', 'ATR', 'ATRA',
                           'ATRC', 'ATRI', 'ATRO', 'ATRS', 'ATSG', 'ATTO', 'ATV', 'ATVI', 'AU', 'AUBN', 'AUDC', 'AUG',
                           'AUMN', 'AUPH', 'AUTO', 'AUY', 'AVA', 'AVAL', 'AVAV', 'AVB', 'AVD', 'AVDL', 'AVEO', 'AVGO',
                           'AVH', 'AVHI', 'AVID', 'AVK', 'AVNW', 'AVP', 'AVT', 'AVX', 'AVXL', 'AVY', 'AWF', 'AWI',
                           'AWK', 'AWP', 'AWR', 'AWRE', 'AWX', 'AXAS', 'AXDX', 'AXE', 'AXGN', 'AXL', 'AXP', 'AXR',
                           'AXS', 'AXSM', 'AXTA', 'AXTI', 'AXU', 'AYI', 'AYR', 'AYTU', 'AZN', 'AZO', 'AZPN', 'AZZ', 'B',
                           'BA', 'BABA', 'BAC', 'BAF', 'BAH', 'BAM', 'BANC', 'BANF', 'BANFP', 'BANR', 'BANX', 'BAP',
                           'BASI', 'BAX', 'BB', 'BBBY', 'BBC', 'BBD', 'BBDO', 'BBF', 'BBGI', 'BBK', 'BBL', 'BBN',
                           'BBOX', 'BBRG', 'BBSI', 'BBVA', 'BBW', 'BBX', 'BBY', 'BC', 'BCBP', 'BCC', 'BCE', 'BCEI',
                           'BCH', 'BCLI', 'BCO', 'BCOM', 'BCOR', 'BCOV', 'BCPC', 'BCRH', 'BCRX', 'BCS', 'BCTF', 'BCV',
                           'BCX', 'BDC', 'BDCZ', 'BDGE', 'BDJ', 'BDL', 'BDN', 'BDR', 'BDSI', 'BDX', 'BEAT', 'BEBE',
                           'BECN', 'BELFA', 'BELFB', 'BEMO', 'BEN', 'BEP', 'BERY', 'BF-A', 'BF-B', 'BFAM', 'BFIN',
                           'BFK', 'BFO', 'BFS', 'BFY', 'BFZ', 'BG', 'BGB', 'BGC', 'BGCP', 'BGFV', 'BGG', 'BGH', 'BGI',
                           'BGNE', 'BGR', 'BGS', 'BGSF', 'BGT', 'BGX', 'BGY', 'BH', 'BHB', 'BHBK', 'BHE', 'BHK', 'BHLB',
                           'BHP', 'BHV', 'BIDU', 'BIF', 'BIG', 'BIIB', 'BIO-B', 'BIO', 'BIOC', 'BIOL', 'BIP', 'BIT',
                           'BITA', 'BJRI', 'BK', 'BKCC', 'BKD', 'BKE', 'BKEP', 'BKEPP', 'BKH', 'BKI', 'BKK', 'BKN',
                           'BKSC', 'BKT', 'BKU', 'BKYI', 'BLBD', 'BLCM', 'BLD', 'BLDP', 'BLDR', 'BLE', 'BLFS', 'BLIN',
                           'BLJ', 'BLK', 'BLKB', 'BLL', 'BLMN', 'BLMT', 'BLPH', 'BLRX', 'BLUE', 'BLW', 'BLX', 'BMA',
                           'BMCH', 'BME', 'BMI', 'BMO', 'BMRA', 'BMRC', 'BMRN', 'BMS', 'BMTC', 'BMY', 'BNED', 'BNFT',
                           'BNS', 'BNSO', 'BNTC', 'BNY', 'BOCH', 'BOE', 'BOKF', 'BOOT', 'BOSC', 'BOSS', 'BOTJ', 'BP',
                           'BPFH', 'BPMC', 'BPMX', 'BPOP', 'BPOPN', 'BPT', 'BPTH', 'BQH', 'BRC', 'BREW', 'BRFS', 'BRG',
                           'BRID', 'BRK-A', 'BRK-B', 'BRKL', 'BRKR', 'BRKS', 'BRN', 'BRO', 'BSBR', 'BSCP', 'BSD', 'BSE',
                           'BSET', 'BSJN', 'BSL', 'BSMX', 'BSPM', 'BSQR', 'BSRR', 'BST', 'BSTC', 'BSX', 'BTA', 'BTI',
                           'BTN', 'BTO', 'BTT', 'BTZ', 'BUD', 'BURL', 'BUSE', 'BVN', 'BVSN', 'BVXV', 'BW', 'BWA',
                           'BWFG', 'BWG', 'BWP', 'BWXT', 'BX', 'BXC', 'BXG', 'BXMT', 'BXP', 'BXS', 'BYBK', 'BYD',
                           'BYFC', 'BYM', 'BZM', 'BZUN', 'C', 'CAA', 'CAAS', 'CABO', 'CAC', 'CACC', 'CACI', 'CAE',
                           'CAF', 'CAG', 'CAH', 'CAI', 'CAJ', 'CAKE', 'CAL', 'CALA', 'CALD', 'CALI', 'CALL', 'CALM',
                           'CALX', 'CAMP', 'CAMT', 'CAPL', 'CAPR', 'CAR', 'CARA', 'CARO', 'CARV', 'CASC', 'CASH',
                           'CASI', 'CASS', 'CASY', 'CAT', 'CATB', 'CATH', 'CATM', 'CATO', 'CATS', 'CAVM', 'CBAY', 'CBB',
                           'CBD', 'CBFV', 'CBH', 'CBI', 'CBIO', 'CBL', 'CBLI', 'CBMG', 'CBPO', 'CBPX', 'CBSH', 'CBSHP',
                           'CBT', 'CBU', 'CC', 'CCBG', 'CCCL', 'CCD', 'CCF', 'CCI', 'CCJ', 'CCK', 'CCL', 'CCLP', 'CCMP',
                           'CCNE', 'CCOI', 'CCRC', 'CCRN', 'CCS', 'CCXI', 'CCZ', 'CDC', 'CDE', 'CDL', 'CDNA', 'CDNS',
                           'CDOR', 'CDR', 'CDTX', 'CDXS', 'CDZI', 'CE', 'CEA', 'CECE', 'CEE', 'CEI', 'CELH', 'CELP',
                           'CEMB', 'CENT', 'CENTA', 'CENX', 'CEO', 'CEQP', 'CERC', 'CERN', 'CERS', 'CET', 'CETV',
                           'CETX', 'CEV', 'CEVA', 'CEZ', 'CF', 'CFA', 'CFG', 'CFMS', 'CFNB', 'CFO', 'CFR', 'CFRX',
                           'CFX', 'CG', 'CGA', 'CGEN', 'CGIX', 'CGNT', 'CGNX', 'CHA', 'CHAD', 'CHCI', 'CHCO', 'CHD',
                           'CHDN', 'CHE', 'CHEF', 'CHEK', 'CHFN', 'CHFS', 'CHGG', 'CHKR', 'CHL', 'CHMA', 'CHMG', 'CHMI',
                           'CHN', 'CHNR', 'CHRS', 'CHRW', 'CHS', 'CHSCL', 'CHSCM', 'CHSCN', 'CHSCO', 'CHT', 'CHTR',
                           'CHU', 'CHUY', 'CHW', 'CHY', 'CI', 'CIA', 'CIB', 'CID', 'CIDM', 'CIEN', 'CIG-C', 'CIG',
                           'CIGI', 'CII', 'CIL', 'CIM', 'CINF', 'CINR', 'CIO', 'CIR', 'CISN', 'CIT', 'CIVB', 'CIX',
                           'CIZ', 'CJJD', 'CKH', 'CKX', 'CL', 'CLB', 'CLBS', 'CLCT', 'CLDT', 'CLF', 'CLI', 'CLIR',
                           'CLM', 'CLNE', 'CLR', 'CLUB', 'CLVS', 'CLWT', 'CLX', 'CM', 'CMCL', 'CMCM', 'CMCO', 'CMCSA',
                           'CMCT', 'CMD', 'CMG', 'CMI', 'CMO', 'CMPR', 'CMRX', 'CMS', 'CMT', 'CMTL', 'CNAT', 'CNBKA',
                           'CNCR', 'CNET', 'CNFR', 'CNHI', 'CNHX', 'CNI', 'CNK', 'CNMD', 'CNO', 'CNOB', 'CNP', 'CNQ',
                           'CNS', 'CNSL', 'CNTY', 'CO', 'COBZ', 'CODI', 'COE', 'COF', 'COG', 'COHN', 'COHR', 'COHU',
                           'COKE', 'COL', 'COLB', 'COLM', 'COM', 'COMM', 'CONE', 'CONN', 'COO', 'COP', 'COR', 'CORE',
                           'CORI', 'CORR', 'COST', 'COT', 'COTY', 'COWN', 'CP', 'CPA', 'CPAC', 'CPB', 'CPE', 'CPG',
                           'CPK', 'CPL', 'CPLA', 'CPLP', 'CPN', 'CPRT', 'CPSH', 'CPSI', 'CPSS', 'CPST', 'CPT', 'CPTA',
                           'CQP', 'CR', 'CRBP', 'CRC', 'CRCM', 'CRD-B', 'CREE', 'CRESY', 'CRF', 'CRH', 'CRHM', 'CRI',
                           'CRIS', 'CRK', 'CRL', 'CRMD', 'CRMT', 'CRNT', 'CRS', 'CRTO', 'CRVL', 'CRVP', 'CRWS', 'CRY',
                           'CRZO', 'CS', 'CSA', 'CSB', 'CSBK', 'CSF', 'CSFL', 'CSGP', 'CSGS', 'CSIQ', 'CSL', 'CSLT',
                           'CSPI', 'CSS', 'CSTE', 'CSV', 'CSWC', 'CSWI', 'CSX', 'CTAS', 'CTB', 'CTEK', 'CTHR', 'CTIB',
                           'CTIC', 'CTL', 'CTLT', 'CTR', 'CTRN', 'CTSH', 'CTSO', 'CTT', 'CTV', 'CTXS', 'CTZ', 'CUB',
                           'CUBA', 'CUBE', 'CUBI', 'CUDA', 'CUK', 'CULP', 'CUTR', 'CUZ', 'CVBF', 'CVCO', 'CVE', 'CVEO',
                           'CVG', 'CVGW', 'CVLY', 'CVM', 'CVRR', 'CVTI', 'CVX', 'CW', 'CWAY', 'CWBC', 'CWCO', 'CWST',
                           'CWT', 'CX', 'CXDC', 'CXH', 'CXO', 'CXP', 'CXSE', 'CXW', 'CYAN', 'CYBE', 'CYCC', 'CYCCP',
                           'CYD', 'CYH', 'CYOU', 'CYRN', 'CYS', 'CYTK', 'CYTR', 'CZNC', 'CZZ', 'D', 'DAC', 'DAKT',
                           'DAL', 'DAN', 'DARE', 'DAX', 'DB', 'DCI', 'DCIX', 'DCO', 'DCP', 'DCT', 'DD', 'DDF', 'DDR',
                           'DDS', 'DDT', 'DDWM', 'DE', 'DEA', 'DECK', 'DEEF', 'DEI', 'DEL', 'DENN', 'DEO', 'DERM',
                           'DEUS', 'DEX', 'DFFN', 'DFND', 'DFP', 'DFS', 'DFVL', 'DFVS', 'DGICA', 'DGICB', 'DGII',
                           'DGLT', 'DGLY', 'DHG', 'DHI', 'DHR', 'DHT', 'DIAX', 'DIN', 'DIS', 'DISCA', 'DISCB', 'DISCK',
                           'DISH', 'DIT', 'DIVY', 'DJCO', 'DK', 'DKL', 'DKS', 'DL', 'DLA', 'DLB', 'DLBL', 'DLBS',
                           'DLHC', 'DLTH', 'DLTR', 'DM', 'DMB', 'DMF', 'DMLP', 'DMO', 'DMPI', 'DMRC', 'DNI', 'DNKN',
                           'DNN', 'DNR', 'DO', 'DOC', 'DOOR', 'DOV', 'DPG', 'DPLO', 'DPST', 'DPW', 'DPZ', 'DQ', 'DRAD',
                           'DRE', 'DRH', 'DRI', 'DRIO', 'DRNA', 'DRRX', 'DS', 'DSE', 'DSGX', 'DSKE', 'DSL', 'DSPG',
                           'DSS', 'DST', 'DSU', 'DSWL', 'DSX', 'DTE', 'DTEA', 'DTQ', 'DTRM', 'DTUL', 'DTUS', 'DTYL',
                           'DTYS', 'DUC', 'DUK', 'DUKH', 'DVA', 'DVAX', 'DVCR', 'DVD', 'DVEM', 'DVN', 'DWAQ', 'DWAS',
                           'DWAT', 'DWCH', 'DWIN', 'DWSN', 'DWTR', 'DX', 'DXB', 'DXC', 'DXCM', 'DXLG', 'DXPE', 'DXR',
                           'DY', 'DYLS', 'DYN', 'DYSL', 'DZSI', 'EA', 'EAB', 'EARN', 'EBAY', 'EBAYL', 'EBF', 'EBIX',
                           'EBMT', 'EBS', 'EBSB', 'EBTC', 'ECC', 'ECF', 'ECHO', 'ECL', 'ECPG', 'ECT', 'ECYT', 'ED',
                           'EDD', 'EDF', 'EDGW', 'EDI', 'EDIT', 'EDU', 'EDUC', 'EE', 'EEA', 'EEFT', 'EEMO', 'EEP',
                           'EEQ', 'EFC', 'EFF', 'EFOI', 'EFR', 'EFX', 'EGBN', 'EGLE', 'EGN', 'EGO', 'EGOV', 'EGRX',
                           'EGY', 'EHI', 'EHTH', 'EIA', 'EIG', 'EIGR', 'EIM', 'EIX', 'EL', 'ELC', 'ELGX', 'ELLO', 'ELP',
                           'ELTK', 'ELU', 'ELY', 'EMAN', 'EMD', 'EMDV', 'EME', 'EMF', 'EMHY', 'EMITF', 'EMJ', 'EMKR',
                           'EML', 'EMMS', 'EMN', 'EMO', 'EMP', 'EMQQ', 'EMR', 'ENB', 'ENLC', 'ENLK', 'ENO', 'ENOR',
                           'ENPH', 'ENR', 'ENS', 'ENSG', 'ENSV', 'ENT', 'ENV', 'EOD', 'EOI', 'EOS', 'EOT', 'EPAY',
                           'EPD', 'EPIX', 'EPM', 'EPR', 'EPZM', 'EQBK', 'EQC', 'EQFN', 'EQM', 'EQR', 'EQS', 'EQT',
                           'ERA', 'ERC', 'ERF', 'ERH', 'ERI', 'ERIC', 'ERJ', 'EROS', 'ES', 'ESBK', 'ESCA', 'ESE',
                           'ESEA', 'ESES', 'ESGR', 'ESNC', 'ESND', 'ESNT', 'ESP', 'ESPR', 'ESRT', 'ESRX', 'ESS', 'ESTE',
                           'ESXB', 'ETB', 'ETFC', 'ETG', 'ETH', 'ETHO', 'ETJ', 'ETM', 'ETN', 'ETO', 'ETP', 'ETR', 'ETV',
                           'ETW', 'ETX', 'ETY', 'EURN', 'EURZ', 'EV', 'EVA', 'EVBN', 'EVC', 'EVF', 'EVG', 'EVGN',
                           'EVHC', 'EVI', 'EVLMC', 'EVN', 'EVO', 'EVOK', 'EVOL', 'EVP', 'EVR', 'EVRI', 'EVTC', 'EVV',
                           'EW', 'EWBC', 'EWGS', 'EWMC', 'EWRE', 'EWUS', 'EXAS', 'EXC', 'EXFO', 'EXG', 'EXK', 'EXLS',
                           'EXP', 'EXPE', 'EXPO', 'EXR', 'EXTN', 'EXTR', 'EYEG', 'EYES', 'EZPW', 'EZT', 'F', 'FAF',
                           'FANG', 'FANH', 'FARM', 'FARO', 'FAST', 'FATE', 'FAX', 'FB', 'FBIO', 'FBIZ', 'FBK', 'FBMS',
                           'FBNC', 'FBR', 'FBSS', 'FC', 'FCAP', 'FCAU', 'FCBC', 'FCCY', 'FCE-A', 'FCEL', 'FCFS', 'FCN',
                           'FCNCA', 'FCO', 'FCVT', 'FDBC', 'FDEU', 'FDS', 'FDUS', 'FEIM', 'FELE', 'FEN', 'FENG',
                           'FET', 'FF', 'FFBC', 'FFC', 'FFIC', 'FFIN', 'FFIV', 'FFKT', 'FFNW', 'FFTY', 'FFWM', 'FHB',
                           'FHN', 'FI', 'FIBK', 'FIBR', 'FICO', 'FII', 'FINL', 'FIS', 'FISK', 'FISV', 'FIT', 'FITB',
                           'FITBI', 'FIVE', 'FIVN', 'FIX', 'FIZZ', 'FL', 'FLC', 'FLDM', 'FLEX', 'FLIC', 'FLL', 'FLO',
                           'FLR', 'FLS', 'FLWS', 'FMBI', 'FMC', 'FMN', 'FMNB', 'FMO', 'FMS', 'FMX', 'FN', 'FNB', 'FNF',
                           'FNG', 'FNHC', 'FNJN', 'FNLC', 'FNV', 'FNWB', 'FOE', 'FOF', 'FOLD', 'FOMX', 'FONR', 'FOR',
                           'FORD', 'FORM', 'FORR', 'FORTY', 'FOSL', 'FPAY', 'FPI', 'FPT', 'FRA', 'FRAN', 'FRBA', 'FRC',
                           'FRD', 'FRME', 'FRO', 'FRPH', 'FRPT', 'FRT', 'FSAM', 'FSBC', 'FSBW', 'FSFG', 'FSI', 'FSLR',
                           'FSM', 'FSP', 'FSS', 'FSTR', 'FSV', 'FTAG', 'FTAI', 'FTEK', 'FTFT', 'FTK', 'FTNT', 'FTR',
                           'FTRI', 'FTS', 'FTVA', 'FUL', 'FULT', 'FUNC', 'FUND', 'FVC', 'FWONA', 'FWONK', 'FWP', 'FWRD',
                           'G', 'GAB', 'GAIA', 'GAIN', 'GALT', 'GAM', 'GAMR', 'GARD', 'GARS', 'GASS', 'GASX', 'GATX',
                           'GBAB', 'GBCI', 'GBDC', 'GBL', 'GBLI', 'GBLIZ', 'GBNK', 'GBR', 'GBT', 'GBX', 'GCAP', 'GCBC',
                           'GCI', 'GCO', 'GCP', 'GCV', 'GDDY', 'GDEN', 'GDL', 'GDOT', 'GE', 'GEB', 'GEC', 'GEF-B',
                           'GEF', 'GEL', 'GEM', 'GEN', 'GENE', 'GEOS', 'GER', 'GERN', 'GES', 'GEVO', 'GF', 'GFF', 'GFI',
                           'GFNCP', 'GFY', 'GGAL', 'GGB', 'GGG', 'GGM', 'GGN', 'GGZ', 'GHC', 'GHL', 'GHY', 'GIII',
                           'GIL', 'GILD', 'GILT', 'GIM', 'GIS', 'GJH', 'GJO', 'GJP', 'GJR', 'GJS', 'GJT', 'GKOS',
                           'GLBR', 'GLBZ', 'GLDD', 'GLNG', 'GLO', 'GLOG', 'GLOP', 'GLP', 'GLPI', 'GLQ', 'GLU', 'GLUU',
                           'GLW', 'GLYC', 'GM', 'GME', 'GMED', 'GMLP', 'GNBC', 'GNC', 'GNCA', 'GNCMA', 'GNE', 'GNK',
                           'GNL', 'GNMK', 'GNRC', 'GNT', 'GNTX', 'GNUS', 'GOEX', 'GOF', 'GOGL', 'GOL', 'GOLD', 'GOOD',
                           'GOOG', 'GOP', 'GORO', 'GPC', 'GPI', 'GPL', 'GPM', 'GPN', 'GPOR', 'GPP', 'GPRK', 'GPT',
                           'GPX', 'GRA', 'GRAM', 'GRBK', 'GRC', 'GRF', 'GRFS', 'GRMN', 'GROW', 'GRP-U', 'GRPN', 'GRUB',
                           'GRVY', 'GRX', 'GS', 'GSAT', 'GSBC', 'GSBD', 'GSEU', 'GSIE', 'GSIT', 'GSK', 'GSL', 'GSS',
                           'GSV', 'GT', 'GTE', 'GTIM', 'GTLS', 'GTN-A', 'GTN', 'GTS', 'GTT', 'GURE', 'GUT', 'GV', 'GVA',
                           'GWB', 'GWGH', 'GWPH', 'GWR', 'GWW', 'GXP', 'GYC', 'H', 'HA', 'HABT', 'HACK', 'HAE', 'HAIN',
                           'HAL', 'HALL', 'HALO', 'HASI', 'HAUD', 'HAWK', 'HAWX', 'HAYN', 'HBAN', 'HBANO', 'HBCP',
                           'HBI', 'HBM', 'HBMD', 'HBP', 'HCA', 'HCAP', 'HCHC', 'HCI', 'HCKT', 'HCM', 'HCOM', 'HDAW',
                           'HDEF', 'HDLV', 'HDP', 'HDS', 'HE', 'HEAR', 'HEES', 'HEI-A', 'HELE', 'HEP', 'HEQ', 'HES',
                           'HEWC', 'HEWI', 'HEWL', 'HEWU', 'HEWY', 'HFC', 'HFXI', 'HFXJ', 'HGH', 'HGSH', 'HHS', 'HI',
                           'HIBB', 'HIE', 'HIG', 'HIHO', 'HIL', 'HIMX', 'HIPS', 'HIX', 'HJPX', 'HJV', 'HL', 'HLF',
                           'HLG', 'HLI', 'HLT', 'HLX', 'HMC', 'HMG', 'HMHC', 'HMNF', 'HMNY', 'HMSY', 'HMTA', 'HMTV',
                           'HMY', 'HNI', 'HNNA', 'HNP', 'HNRG', 'HNW', 'HOFT', 'HOG', 'HOLI', 'HOLX', 'HOML', 'HOPE',
                           'HOS', 'HOV', 'HOVNP', 'HP', 'HPE', 'HPF', 'HPI', 'HPP', 'HPQ', 'HPS', 'HQCL', 'HQH', 'HQL',
                           'HQY', 'HR', 'HRB', 'HRC', 'HRI', 'HRL', 'HRTG', 'HRZN', 'HSBC', 'HSC', 'HSEB', 'HSIC',
                           'HSII', 'HSKA', 'HSON', 'HST', 'HSTM', 'HSY', 'HTBI', 'HTGC', 'HTGM', 'HTH', 'HTHT', 'HTLF',
                           'HTUS', 'HTZ', 'HUBB', 'HUBG', 'HUBS', 'HUM', 'HUN', 'HURC', 'HURN', 'HUSA', 'HVT-A', 'HVT',
                           'HWBK', 'HWKN', 'HYHG', 'HYI', 'HYXU', 'HZNP', 'HZO', 'I', 'IAC', 'IAE', 'IAF', 'IAG',
                           'IAGG', 'IAM', 'IART', 'IBIO', 'IBKC', 'IBKCP', 'IBKR', 'IBMJ', 'IBMK', 'IBN', 'IBOC', 'IBP',
                           'IBTX', 'ICAD', 'ICB', 'ICBK', 'ICCC', 'ICD', 'ICE', 'ICFI', 'ICL', 'ICLR', 'ICPT', 'IDA',
                           'IDCC', 'IDLB', 'IDN', 'IDRA', 'IDTI', 'IDXG', 'IDXX', 'IEC', 'IEP', 'IESC', 'IEX', 'IF',
                           'IFLY', 'IFN', 'IGA', 'IGC', 'IGD', 'IGHG', 'IGR', 'IHC', 'IHG', 'IHT', 'IID', 'IIF', 'III',
                           'IIIN', 'IIM', 'IIN', 'IIVI', 'IKNX', 'ILG', 'ILMN', 'IMAX', 'IMDZ', 'IMGN', 'IMH', 'IMMR',
                           'IMMU', 'IMO', 'IMOM', 'IMUC', 'INAP', 'INBK', 'INCY', 'INDB', 'INF', 'INFI', 'INFN', 'INFO',
                           'INFU', 'INFY', 'ING', 'INGN', 'INGR', 'INO', 'INOD', 'INPX', 'INS', 'INSG', 'INSI', 'INSM',
                           'INST', 'INTC', 'INTG', 'INTL', 'INTT', 'INTU', 'INTX', 'INVE', 'INWK', 'IO', 'IOR', 'IOSP',
                           'IOTS', 'IP', 'IPAR', 'IPAS', 'IPAY', 'IPB', 'IPG', 'IPHI', 'IPHS', 'IPOS', 'IPWR', 'IPXL',
                           'IQDG', 'IQI', 'IR', 'IRCP', 'IRDM', 'IRIX', 'IRM', 'IROQ', 'IRR', 'IRS', 'IRT', 'IRWD',
                           'ISBC', 'ISD', 'ISG', 'ISIG', 'ISL', 'ISNS', 'ISR', 'ISSC', 'ISTR', 'ISZE', 'IT', 'ITCB',
                           'ITCI', 'ITIC', 'ITRN', 'ITT', 'ITUB', 'ITW', 'IVAC', 'IVAL', 'IVC', 'IVLU', 'IX', 'IYLD',
                           'JAG', 'JAKK', 'JASN', 'JAX', 'JBK', 'JBL', 'JBLU', 'JBN', 'JBSS', 'JBT', 'JCAP', 'JCE',
                           'JCI', 'JCO', 'JCOM', 'JCP', 'JCTCF', 'JD', 'JE', 'JEC', 'JEQ', 'JFR', 'JGH', 'JHI', 'JHMC',
                           'JHMF', 'JHMH', 'JHML', 'JHMM', 'JHMS', 'JHMT', 'JHS', 'JHX', 'JHY', 'JJSF', 'JLS', 'JMBA',
                           'JMEI', 'JMLP', 'JMM', 'JNJ', 'JNPR', 'JOB', 'JOF', 'JOUT', 'JP', 'JPEM', 'JPM', 'JPN',
                           'JPS', 'JPT', 'JPUS', 'JPXN', 'JQC', 'JRI', 'JRS', 'JRVR', 'JSD', 'JSM', 'JSMD', 'JSML',
                           'JTA', 'JTD', 'JTPY', 'JW-A', 'JW-B', 'JWN', 'JYNT', 'KAI', 'KALU', 'KALV', 'KAMN', 'KAR',
                           'KB', 'KBAL', 'KBR', 'KBSF', 'KBWB', 'KBWD', 'KBWY', 'KE', 'KED', 'KELYA', 'KELYB', 'KEM',
                           'KEP', 'KEQU', 'KEX', 'KEY', 'KFFB', 'KGC', 'KHC', 'KIM', 'KIN', 'KINS', 'KL', 'KLAC',
                           'KLDW', 'KLDX', 'KLIC', 'KLXI', 'KMB', 'KMDA', 'KMF', 'KMG', 'KMM', 'KMT', 'KMX', 'KN',
                           'KND', 'KNDI', 'KNL', 'KNX', 'KO', 'KODK', 'KOF', 'KOP', 'KOPN', 'KOS', 'KOSS', 'KPTI',
                           'KRC', 'KRG', 'KRNT', 'KRO', 'KS', 'KSS', 'KST', 'KTEC', 'KTF', 'KTH', 'KTN', 'KTOV', 'KTP',
                           'KTWO', 'KURA', 'KW', 'KWEB', 'KWR', 'KYN', 'L', 'LADR', 'LAKE', 'LAMR', 'LANC', 'LAND',
                           'LAQ', 'LARK', 'LAWS', 'LAZ', 'LB', 'LBAI', 'LBDC', 'LBRDA', 'LBRDK', 'LBTYA', 'LBTYB',
                           'LBTYK', 'LBY', 'LCII', 'LCNB', 'LCUT', 'LDL', 'LDOS', 'LDP', 'LE', 'LEA', 'LEAD', 'LECO',
                           'LEDS', 'LEE', 'LEG', 'LEJU', 'LEN-B', 'LEN', 'LEO', 'LFC', 'LFUS', 'LGF-A', 'LGI', 'LGIH',
                           'LGND', 'LH', 'LHCG', 'LHO', 'LIFE', 'LILA', 'LILAK', 'LINC', 'LIND', 'LINK', 'LIQT', 'LITB',
                           'LITE', 'LIVE', 'LJPC', 'LKFN', 'LKOR', 'LLEX', 'LLIT', 'LLNW', 'LLY', 'LMAT', 'LMHA',
                           'LMNX', 'LMRK', 'LMT', 'LN', 'LNC', 'LNCE', 'LND', 'LNG', 'LNT', 'LNTH', 'LOAN', 'LOCO',
                           'LOGI', 'LOGM', 'LOPE', 'LOV', 'LOW', 'LPCN', 'LPG', 'LPL', 'LPLA', 'LPNT', 'LPSN', 'LPTH',
                           'LPX', 'LQ', 'LRCX', 'LRET', 'LSCC', 'LSI', 'LSXMA', 'LTBR', 'LTC', 'LTM', 'LTRPA', 'LUNA',
                           'LUV', 'LVNTB', 'LVS', 'LWAY', 'LXP', 'LXU', 'LYB', 'LYTS', 'LYV', 'LZB', 'M', 'MA', 'MAB',
                           'MAGS', 'MAIN', 'MAN', 'MANH', 'MANT', 'MANU', 'MAR', 'MARA', 'MARK', 'MAS', 'MASI', 'MAT',
                           'MATR', 'MATX', 'MAV', 'MAXR', 'MAYS', 'MBCN', 'MBFI', 'MBI', 'MBII', 'MBOT', 'MBUU', 'MBWM',
                           'MC', 'MCA', 'MCC', 'MCD', 'MCF', 'MCFT', 'MCHP', 'MCHX', 'MCI', 'MCK', 'MCN', 'MCO', 'MCR',
                           'MCRB', 'MCS', 'MCV', 'MCY', 'MD', 'MDC', 'MDGL', 'MDGS', 'MDLY', 'MDP', 'MDRX', 'MDT',
                           'MDU', 'MDWD', 'MEAR', 'MED', 'MEI', 'MEIP', 'MELI', 'MELR', 'MEOH', 'MERC', 'MESO', 'MFA',
                           'MFC', 'MFG', 'MFIN', 'MFL', 'MFM', 'MFNC', 'MFO', 'MFSF', 'MFT', 'MFV', 'MG', 'MGEE',
                           'MGEN', 'MGF', 'MGI', 'MGM', 'MGNX', 'MGP', 'MGPI', 'MGYR', 'MHD', 'MHF', 'MHH', 'MHI',
                           'MHLA', 'MHLD', 'MHNC', 'MHO', 'MICR', 'MICT', 'MIE', 'MIN', 'MIND', 'MINI', 'MITK', 'MITL',
                           'MITT', 'MIW', 'MKC-V', 'MKC', 'MKSI', 'MKTX', 'MLCO', 'MLHR', 'MLI', 'MLM', 'MLNX', 'MLR',
                           'MLSS', 'MLTI', 'MLVF', 'MMAC', 'MMD', 'MMI', 'MMLP', 'MMM', 'MMP', 'MMS', 'MMSI', 'MMT',
                           'MMU', 'MMV', 'MMYT', 'MN', 'MNDO', 'MNK', 'MNKD', 'MNOV', 'MNP', 'MNR', 'MNRO', 'MNST',
                           'MNTA', 'MO', 'MOBL', 'MOD', 'MOG-A', 'MOG-B', 'MOH', 'MOMO', 'MON', 'MORN', 'MOS', 'MOSY',
                           'MOTI', 'MOV', 'MOXC', 'MPA', 'MPAA', 'MPB', 'MPC', 'MPLX', 'MPVD', 'MPW', 'MPWR', 'MQT',
                           'MQY', 'MRCY', 'MRK', 'MRLN', 'MRNS', 'MRO', 'MRRL', 'MRTX', 'MRVL', 'MSB', 'MSBF', 'MSCI',
                           'MSD', 'MSEX', 'MSFT', 'MSGN', 'MSI', 'MSM', 'MSN', 'MSTR', 'MT', 'MTB', 'MTBC', 'MTBCP',
                           'MTCH', 'MTD', 'MTDR', 'MTEX', 'MTG', 'MTGE', 'MTH', 'MTL', 'MTLS', 'MTN', 'MTNB', 'MTR',
                           'MTRN', 'MTRX', 'MTSC', 'MTSI', 'MTT', 'MTX', 'MTZ', 'MU', 'MUA', 'MUE', 'MUH', 'MUI', 'MUJ',
                           'MUS', 'MUSA', 'MUX', 'MVC', 'MVF', 'MVIN', 'MVO', 'MVT', 'MWA', 'MX', 'MXC', 'MXF', 'MXIM',
                           'MXL', 'MYE', 'MYGN', 'MYI', 'MYJ', 'MYL', 'MYN', 'MYOK', 'MYOS', 'MYRG', 'MZA', 'NAC',
                           'NAD', 'NAIL', 'NAKD', 'NAN', 'NANR', 'NAOV', 'NAP', 'NAT', 'NATH', 'NAUH', 'NAV', 'NAVB',
                           'NAVI', 'NAZ', 'NBB', 'NBEV', 'NBH', 'NBIX', 'NBN', 'NBO', 'NBRV', 'NBW', 'NBY', 'NC', 'NCA',
                           'NCB', 'NCLH', 'NCOM', 'NCR', 'NCV', 'NCZ', 'NDLS', 'NDSN', 'NE', 'NEA', 'NEN', 'NEO',
                           'NEON', 'NEOS', 'NEPT', 'NERV', 'NEU', 'NEV', 'NEWR', 'NEWT', 'NEXT', 'NFBK', 'NFJ', 'NFX',
                           'NGD', 'NGHC', 'NGHCP', 'NGL', 'NGVC', 'NHA', 'NHC', 'NHLD', 'NHTC', 'NI', 'NICE', 'NID',
                           'NIE', 'NIHD', 'NIM', 'NIQ', 'NJR', 'NK', 'NKE', 'NKG', 'NKSH', 'NKTR', 'NL', 'NLST', 'NLY',
                           'NMFC', 'NMI', 'NMIH', 'NML', 'NMM', 'NMR', 'NMS', 'NMY', 'NMZ', 'NNBR', 'NNN', 'NNVC',
                           'NOAH', 'NOG', 'NOM', 'NOMD', 'NOVT', 'NOW', 'NP', 'NPK', 'NPN', 'NPO', 'NPTN', 'NRCIB',
                           'NRG', 'NRIM', 'NRK', 'NRO', 'NRP', 'NRT', 'NRZ', 'NS', 'NSC', 'NSEC', 'NSH', 'NSM', 'NSP',
                           'NSPR', 'NSSC', 'NSTG', 'NSYS', 'NTAP', 'NTB', 'NTCT', 'NTEC', 'NTES', 'NTG', 'NTGR', 'NTIC',
                           'NTIP', 'NTL', 'NTN', 'NTP', 'NTRA', 'NTRI', 'NTRS', 'NTWK', 'NTZ', 'NUE', 'NUM', 'NUS',
                           'NUSA', 'NUV', 'NUVA', 'NVAX', 'NVCN', 'NVEC', 'NVG', 'NVGS', 'NVIV', 'NVO', 'NVRO', 'NVS',
                           'NVTA', 'NVUS', 'NWBI', 'NWE', 'NWHM', 'NWL', 'NWLI', 'NWPX', 'NWS', 'NWSA', 'NXC', 'NXJ',
                           'NXN', 'NXQ', 'NXRT', 'NXST', 'NXTD', 'NYH', 'NYMT', 'NYMTO', 'NYMX', 'NYRT', 'NYV', 'NZF',
                           'O', 'OBE', 'OC', 'OCFC', 'OCIP', 'OCLR', 'OCN', 'OCSI', 'OCSL', 'OCUL', 'ODC', 'ODP', 'OEC',
                           'OESX', 'OEUR', 'OFC', 'OFED', 'OFG', 'OFIX', 'OFLX', 'OGCP', 'OGE', 'OGEN', 'OHI', 'OI',
                           'OIBR-C', 'OII', 'OIIM', 'OKE', 'OLED', 'OLLI', 'OLN', 'OLP', 'OMER', 'OMEX', 'OMF', 'OMI',
                           'OMN', 'ONB', 'ONCS', 'ONDK', 'ONEO', 'ONEV', 'ONEY', 'ONTX', 'ONVO', 'OOMA', 'OPHC', 'OPK',
                           'OPNT', 'OPOF', 'OPP', 'OPTT', 'OPY', 'OR', 'ORA', 'ORAN', 'ORBC', 'ORC', 'ORCL', 'ORG',
                           'ORIG', 'ORLY', 'ORMP', 'OSB', 'OSBCP', 'OSG', 'OSIS', 'OSN', 'OSTK', 'OSUR', 'OTEL', 'OTEX',
                           'OTIC', 'OTTW', 'OUSA', 'OUT', 'OVBC', 'OVLY', 'OXBR', 'OXLC', 'OXLCO', 'OXY', 'P', 'PAAS',
                           'PACW', 'PAHC', 'PAM', 'PANL', 'PANW', 'PAR', 'PARR', 'PATI', 'PATK', 'PAYX', 'PBB', 'PBBI',
                           'PBCT', 'PBF', 'PBH', 'PBIO', 'PBIP', 'PBPB', 'PBR', 'PBSK', 'PBT', 'PBYI', 'PCF', 'PCG',
                           'PCH', 'PCI', 'PCK', 'PCM', 'PCN', 'PCOM', 'PCQ', 'PCRX', 'PCTI', 'PCTY', 'PCYG', 'PCYO',
                           'PDCO', 'PDEX', 'PDFS', 'PDI', 'PDLI', 'PDM', 'PDS', 'PDT', 'PE', 'PEBK', 'PEBO', 'PED',
                           'PEG', 'PEGA', 'PEGI', 'PEI', 'PEIX', 'PEN', 'PENN', 'PEO', 'PEP', 'PER', 'PERY', 'PESI',
                           'PF', 'PFBC', 'PFBI', 'PFBX', 'PFE', 'PFG', 'PFH', 'PFI', 'PFIE', 'PFIN', 'PFL', 'PFLT',
                           'PFNX', 'PFS', 'PFSI', 'PGC', 'PGEM', 'PGH', 'PGP', 'PGR', 'PGRE', 'PGZ', 'PH', 'PHD', 'PHH',
                           'PHI', 'PHK', 'PHM', 'PHO', 'PHX', 'PHYS', 'PICO', 'PID', 'PIH', 'PINC', 'PIR', 'PIRS',
                           'PIY', 'PIZ', 'PJH', 'PJT', 'PKE', 'PKO', 'PKOH', 'PKW', 'PKX', 'PLAB', 'PLAY', 'PLBC',
                           'PLCE', 'PLD', 'PLG', 'PLM', 'PLOW', 'PLPC', 'PLT', 'PLUG', 'PLUS', 'PLW', 'PLX', 'PLXP',
                           'PLXS', 'PLYA', 'PM', 'PMD', 'PME', 'PMF', 'PML', 'PMM', 'PMO', 'PMX', 'PNF', 'PNFP', 'PNI',
                           'PNM', 'PNNT', 'PNR', 'PNW', 'PODD', 'POL', 'POOL', 'POPE', 'POR', 'POST', 'POWI', 'POWL',
                           'PPC', 'PPIH', 'PPR', 'PPT', 'PPX', 'PRA', 'PRAA', 'PRAH', 'PRCP', 'PRFT', 'PRGS', 'PRGX',
                           'PRH', 'PRI', 'PRIM', 'PRK', 'PRKR', 'PRLB', 'PRME', 'PRMW', 'PRO', 'PROV', 'PRPH', 'PRQR',
                           'PRSS', 'PRTA', 'PRTK', 'PRTS', 'PRTY', 'PRU', 'PSA', 'PSB', 'PSET', 'PSF', 'PSL', 'PSMT',
                           'PSO', 'PSTG', 'PSTI', 'PSX', 'PSXP', 'PTCT', 'PTEU', 'PTF', 'PTH', 'PTI', 'PTLA', 'PTLC',
                           'PTNQ', 'PTNR', 'PTR', 'PTY', 'PUK', 'PUTW', 'PVBC', 'PVG', 'PVH', 'PW', 'PWR', 'PXD', 'PXI',
                           'PXLW', 'PXS', 'PYN', 'PYS', 'PYT', 'PZC', 'PZE', 'PZG', 'PZN', 'PZZA', 'QADA', 'QCOM',
                           'QCRH', 'QDEL', 'QGEN', 'QGTA', 'QIWI', 'QLC', 'QLYS', 'QMOM', 'QNST', 'QQQX', 'QRHC',
                           'QRVO', 'QSR', 'QTNT', 'QTRH', 'QTS', 'QTWO', 'QUAD', 'QUIK', 'QUMU', 'QUOT', 'QVAL', 'QVM',
                           'R', 'RACE', 'RAD', 'RADA', 'RAND', 'RARE', 'RBA', 'RBCAA', 'RBCN', 'RBS', 'RCG', 'RCI',
                           'RCII', 'RCKY', 'RCL', 'RCMT', 'RCON', 'RCS', 'RDCM', 'RDHL', 'RDI', 'RDIB', 'RDN', 'RDNT',
                           'RDS-A', 'RDS-B', 'RDUS', 'RDWR', 'RDY', 'RE', 'RECN', 'REG', 'REGI', 'REGN', 'REI', 'REIS',
                           'RELV', 'RENN', 'RENX', 'REPH', 'RESI', 'RESN', 'REV', 'REX', 'RF', 'RFI', 'RFIL', 'RFP',
                           'RGA', 'RGC', 'RGCO', 'RGEN', 'RGLD', 'RGLS', 'RGNX', 'RGR', 'RGS', 'RGT', 'RH', 'RHI',
                           'RHP', 'RIBT', 'RIC', 'RICK', 'RIF', 'RIG', 'RIGL', 'RIO', 'RISE', 'RIV', 'RJF', 'RL',
                           'RLGT', 'RLGY', 'RLH', 'RLI', 'RLJ', 'RLJE', 'RM', 'RMAX', 'RMBS', 'RMCF', 'RMD', 'RMNI',
                           'RMNIU', 'RMT', 'RMTI', 'RNET', 'RNG', 'RNWK', 'ROAM', 'RODM', 'ROG', 'ROGS', 'ROIC', 'ROK',
                           'ROL', 'ROLL', 'ROP', 'ROSE', 'ROST', 'ROUS', 'ROYT', 'RP', 'RPAI', 'RPD', 'RPM', 'RPT',
                           'RPXC', 'RQI', 'RRC', 'RRR', 'RRTS', 'RS', 'RSLS', 'RSPP', 'RTIX', 'RTN', 'RTRX', 'RTTR',
                           'RUBI', 'RUN', 'RUSHA', 'RUTH', 'RVLT', 'RVNC', 'RVP', 'RVT', 'RXDX', 'RXN', 'RY', 'RYAM',
                           'RYN', 'S', 'SA', 'SABR', 'SAFT', 'SAGE', 'SAH', 'SAIA', 'SAL', 'SALM', 'SALT', 'SAMG',
                           'SAND', 'SANM', 'SAR', 'SASR', 'SATS', 'SAVE', 'SB', 'SBBX', 'SBCF', 'SBFG', 'SBGI', 'SBI',
                           'SBLK', 'SBNY', 'SBRA', 'SBS', 'SBSI', 'SBT', 'SBUX', 'SC', 'SCCO', 'SCD', 'SCHN', 'SCHW',
                           'SCI', 'SCKT', 'SCL', 'SCM', 'SCMP', 'SCON', 'SCS', 'SCSC', 'SCVL', 'SCX', 'SCYX', 'SDRL',
                           'SEAC', 'SEB', 'SEIC', 'SELF', 'SENEA', 'SENEB', 'SENS', 'SEP', 'SERV', 'SF', 'SFBC', 'SFBS',
                           'SFL', 'SFM', 'SFNC', 'SFUN', 'SGA', 'SGC', 'SGMA', 'SGMO', 'SGOC', 'SGRP', 'SGRY', 'SGU',
                           'SHAK', 'SHE', 'SHEN', 'SHG', 'SHIP', 'SHLM', 'SHLO', 'SHO', 'SHOO', 'SHOP', 'SHPG', 'SHSP',
                           'SHW', 'SID', 'SIEB', 'SIF', 'SIFY', 'SIG', 'SIGI', 'SIGM', 'SILC', 'SIM', 'SIMO', 'SINA',
                           'SINO', 'SIR', 'SIRI', 'SJI', 'SJM', 'SJT', 'SKM', 'SKY', 'SKYS', 'SLAB', 'SLB', 'SLCA',
                           'SLCT', 'SLF', 'SLGN', 'SLM', 'SLMBP', 'SLNO', 'SLP', 'SLRC', 'SM', 'SMBC', 'SMCI', 'SMED',
                           'SMFG', 'SMG', 'SMHD', 'SMIN', 'SMIT', 'SMM', 'SMMF', 'SMMT', 'SMP', 'SMSI', 'SMTC', 'SMTS',
                           'SNA', 'SNBR', 'SNCR', 'SNDX', 'SNE', 'SNFCA', 'SNI', 'SNMP', 'SNMX', 'SNN', 'SNOA', 'SNP',
                           'SNPS', 'SNR', 'SNV', 'SNX', 'SO', 'SODA', 'SOHO', 'SOHU', 'SOJA', 'SONA', 'SOR', 'SOVB',
                           'SPAR', 'SPB', 'SPCB', 'SPDW', 'SPEM', 'SPGI', 'SPH', 'SPHS', 'SPIB', 'SPIL', 'SPKE', 'SPLG',
                           'SPLK', 'SPLP', 'SPMD', 'SPMO', 'SPNE', 'SPNS', 'SPOK', 'SPPI', 'SPPP', 'SPR', 'SPRT',
                           'SPSC', 'SPTM', 'SPTN', 'SPTS', 'SPWR', 'SPXC', 'SPXE', 'SPXN', 'SPXV', 'SPYD', 'SQ', 'SQM',
                           'SQNS', 'SR', 'SRAX', 'SRC', 'SRCL', 'SRDX', 'SRE', 'SREV', 'SRI', 'SRLP', 'SRNE', 'SRPT',
                           'SRT', 'SRV', 'SSB', 'SSBI', 'SSC', 'SSD', 'SSI', 'SSKN', 'SSL', 'SSNC', 'SSNT', 'SSP',
                           'SSRM', 'SSTK', 'SSY', 'SSYS', 'STAA', 'STAG', 'STAR', 'STAY', 'STBA', 'STC', 'STE', 'STI',
                           'STK', 'STKL', 'STKS', 'STLY', 'STM', 'STN', 'STNG', 'STOT', 'STPP', 'STRA', 'STRL', 'STRM',
                           'STRP', 'STRS', 'STT', 'STWD', 'STZ-B', 'STZ', 'SU', 'SUI', 'SUM', 'SUMR', 'SUN', 'SUNS',
                           'SUNW', 'SUP', 'SUPN', 'SUSA', 'SVA', 'SVBI', 'SVT', 'SVU', 'SVVC', 'SWIR', 'SWK', 'SWKS',
                           'SWM', 'SWN', 'SWX', 'SWZ', 'SXC', 'SYBX', 'SYK', 'SYKE', 'SYN', 'SYNA', 'SYNC', 'SYNL',
                           'SYNT', 'SYPR', 'SYX', 'SYY', 'SZC', 'T', 'TAC', 'TACO', 'TAIT', 'TANH', 'TANNL', 'TANNZ',
                           'TARO', 'TAST', 'TAT', 'TATT', 'TAYD', 'TBB', 'TBK', 'TBNK', 'TBPH', 'TCBI', 'TCBIL', 'TCBK',
                           'TCCO', 'TCF', 'TCFC', 'TCI', 'TCO', 'TCPC', 'TCS', 'TCX', 'TD', 'TDA', 'TDC', 'TDE', 'TDF',
                           'TDI', 'TDJ', 'TDOC', 'TDW', 'TEAM', 'TECD', 'TECH', 'TECK', 'TEDU', 'TELL', 'TEN', 'TENX',
                           'TEP', 'TER', 'TERP', 'TESS', 'TEUM', 'TEVA', 'TFX', 'TGA', 'TGB', 'TGC', 'TGEN', 'TGH',
                           'TGI', 'TGLS', 'TGNA', 'TGP', 'TGS', 'TGT', 'TGTX', 'THC', 'THG', 'THGA', 'THM', 'THO',
                           'THRM', 'THS', 'THST', 'THW', 'TIF', 'TIPT', 'TISI', 'TITN', 'TIVO', 'TJX', 'TK', 'TKAT',
                           'TKC', 'TKR', 'TLDH', 'TLEH', 'TLF', 'TLGT', 'TLI', 'TLK', 'TLRA', 'TLRD', 'TLYS', 'TM',
                           'TMP', 'TMQ', 'TMST', 'TMUS', 'TNAV', 'TNDM', 'TNET', 'TNH', 'TNP', 'TNXP', 'TOO', 'TOPS',
                           'TOUR', 'TPC', 'TPH', 'TPHS', 'TPL', 'TPR', 'TPVG', 'TPYP', 'TPZ', 'TR', 'TRC', 'TREC',
                           'TREE', 'TREX', 'TRGP', 'TRI', 'TRIB', 'TRIP', 'TRMB', 'TRMK', 'TRN', 'TRNO', 'TRNS', 'TROV',
                           'TROW', 'TROX', 'TRP', 'TRQ', 'TRS', 'TRST', 'TRT', 'TRTN', 'TRU', 'TRUE', 'TRUP', 'TRV',
                           'TRVN', 'TRX', 'TRXC', 'TS', 'TSBK', 'TSC', 'TSCO', 'TSE', 'TSEM', 'TSG', 'TSI', 'TSLA',
                           'TSLX', 'TSM', 'TSN', 'TSRO', 'TSU', 'TTC', 'TTEC', 'TTEK', 'TTGT', 'TTI', 'TTM', 'TTMI',
                           'TTP', 'TTWO', 'TU', 'TUES', 'TUP', 'TURN', 'TV', 'TVC', 'TVE', 'TVPT', 'TVTY', 'TWI',
                           'TWLO', 'TWMC', 'TWN', 'TWNK', 'TWO', 'TWOU', 'TWTR', 'TX', 'TXMD', 'TXN', 'TY', 'TYG',
                           'TYL', 'TYME', 'TZOO', 'UA', 'UAA', 'UAL', 'UAN', 'UBA', 'UBCP', 'UBIO', 'UBOH', 'UBS',
                           'UBSI', 'UCBA', 'UCBI', 'UCFC', 'UEC', 'UEIC', 'UFCS', 'UFI', 'UFPI', 'UFPT', 'UFS', 'UG',
                           'UGI', 'UGP', 'UHAL', 'UHS', 'UHT', 'UL', 'ULTA', 'UMC', 'UMH', 'UMPQ', 'UN', 'UNB', 'UNM',
                           'UNP', 'UNT', 'UNTY', 'UNVR', 'UONEK', 'UPL', 'UPS', 'URBN', 'URG', 'URI', 'USA', 'USAC',
                           'USAK', 'USAP', 'USAS', 'USAT', 'USATP', 'USB', 'USCR', 'USEG', 'USLB', 'USLM', 'USM',
                           'USNA', 'USRT', 'UTES', 'UTF', 'UTG', 'UTHR', 'UTI', 'UTL', 'UTMD', 'UTSI', 'UUU', 'UUUU',
                           'UVSP', 'UVV', 'UZA', 'UZB', 'V', 'VAC', 'VALE', 'VALU', 'VALX', 'VAR', 'VBF', 'VBFC',
                           'VBIV', 'VBLT', 'VBND', 'VC', 'VCEL', 'VCF', 'VCYT', 'VEC', 'VECO', 'VEDL', 'VEEV', 'VEON',
                           'VER', 'VERU', 'VET', 'VFC', 'VFL', 'VG', 'VGI', 'VGM', 'VGZ', 'VHC', 'VHI', 'VIAV', 'VICR',
                           'VIPS', 'VIRT', 'VIV', 'VIVE', 'VIVO', 'VJET', 'VKI', 'VKQ', 'VKTX', 'VLGEA', 'VLO', 'VLP',
                           'VLRS', 'VLY', 'VMC', 'VMI', 'VMO', 'VMW', 'VNCE', 'VNET', 'VNO', 'VNOM', 'VOC', 'VOXX',
                           'VR', 'VRA', 'VRML', 'VRNS', 'VRNT', 'VRSK', 'VRSN', 'VRTS', 'VRTU', 'VRTV', 'VRTX', 'VSEC',
                           'VSH', 'VSLR', 'VST', 'VSTM', 'VSTO', 'VTA', 'VTEB', 'VTN', 'VTNR', 'VTR', 'VTVT', 'VVI',
                           'VYGR', 'VYMI', 'VZA', 'W', 'WAB', 'WAFD', 'WAL', 'WASH', 'WAT', 'WATT', 'WBA', 'WBBW',
                           'WBIA', 'WBIE', 'WBIF', 'WBIG', 'WBIH', 'WBII', 'WBIL', 'WBK', 'WBS', 'WBT', 'WCC', 'WCFB',
                           'WCN', 'WDAY', 'WDC', 'WDFC', 'WDR', 'WEA', 'WEB', 'WEBK', 'WEN', 'WERN', 'WETF', 'WEX',
                           'WFC', 'WGL', 'WGO', 'WHF', 'WHG', 'WHLM', 'WHLR', 'WHR', 'WIA', 'WIFI', 'WINA', 'WING',
                           'WINS', 'WIT', 'WIW', 'WKHS', 'WLDN', 'WLFC', 'WLL', 'WMC', 'WMGI', 'WMS', 'WMT', 'WNC',
                           'WNEB', 'WNS', 'WOR', 'WPC', 'WPG', 'WPRT', 'WPX', 'WPZ', 'WRE', 'WRI', 'WRK', 'WRLD',
                           'WSBC', 'WSCI', 'WSM', 'WSO-B', 'WSO', 'WSR', 'WST', 'WTBA', 'WTFC', 'WTFCM', 'WTI', 'WTM',
                           'WTR', 'WTS', 'WUBA', 'WVE', 'WVFC', 'WVVI', 'WWD', 'WWE', 'WWR', 'WWW', 'WY', 'WYY', 'X',
                           'XBIT', 'XCEM', 'XCRA', 'XEC', 'XEL', 'XELB', 'XENT', 'XHR', 'XIN', 'XITK', 'XL', 'XLNX',
                           'XLRE', 'XLRN', 'XNCR', 'XNET', 'XNTK', 'XOM', 'XOMA', 'XONE', 'XOXO', 'XPER', 'XPL', 'XPO',
                           'XRAY', 'XRM', 'XTLB', 'XXII', 'XYL', 'Y', 'YELP', 'YLD', 'YNDX', 'YORW', 'YRCW', 'YRD',
                           'YTEN', 'YUM', 'YY', 'Z', 'ZAIS', 'ZAYO', 'ZBH', 'ZBIO', 'ZBK', 'ZBRA', 'ZEN', 'ZGNX',
                           'ZION', 'ZIOP', 'ZIXI', 'ZN', 'ZNH', 'ZOES', 'ZSAN', 'ZTR', 'ZTS', 'ZYNE']

        self.__tickers_array_short = tickerlist = ['MSFT', 'AAPL', 'AMZN', 'FB', 'GOOGL', 'TSLA',
              'INTC', 'NVDA', 'NFLX', 'ADBE', 'PYPL', 'CSCO', 'PEP', 'AROW',
              'CMCSA', 'AMGN', 'COST', 'TMUS', 'AVGO', 'TXN', 'CHTR', 'ABEV',
              'QCOM', 'GILD', 'SBUX', 'INTU', 'VRTX', 'MDLZ', 'BKNG',
              'ISRG', 'FISV', 'REGN', 'ADP', 'AMD', 'ATVI', 'JD', 'AMAT',
              'ILMN', 'MU', 'CSX', 'ADSK', 'MELI', 'LRCX', 'ADI', 'DOV',
              'BIIB', 'EBAY', 'DXCM', 'KHC', 'EA', 'LULU', 'MNST', 'WBA',
              'EXC', 'BIDU', 'XEL', 'WDAY', 'NTES', 'NXPI', 'VFC', 'FMC',
              'KLAC', 'ORLY', 'SPLK', 'ROST', 'SGEN', 'CTSH', 'SNPS', 'HSIC',
              'ASML', 'IDXX', 'MAR', 'CSGP', 'CTAS', 'VRSK', 'CDNS', 'GFI',
              'PAYX', 'PCAR', 'MCHP', 'ANSS', 'SIRI', 'FAST', 'ALXN', 'CBSH',
              'VRSN', 'XLNX', 'INCY', 'BMRN', 'SWKS', 'ALGN', 'DLTR', 'RYN', 'MDC',
              'CPRT', 'CTXS', 'TTWO', 'CHKP', 'MXIM', 'CDW','TCOM', 'NAT', 'CERN',
              'WDC', 'EXPE', 'ULTA', 'NTAP', 'LBTYK', 'ASH', 'EQT', 'TSM', 'APD',
              'LBTYA', 'HPQ', 'DXC', 'CAG','MAS', 'SPXC', 'BAM', 'TGNA', 'VAR', 'SNN']


    def get_tickers(self, list_number):
        if(list_number == 1):
            return self.__tickers_array
        elif (list_number == 2):
            return self.__tickers_array_short

    def __get_tickers(self, subdir):

        return [re.findall(r'.*_(.*)\.\w{3}', f.name)[0] for f in os.scandir(self.__fileDir+ subdir) if f.is_file()]

    def __list_tickers(self):

        dirOutput = '/data/stocks/'
        array = []
        for __ticker in self.__get_tickers(dirOutput):
            array.append(__ticker)

        print(array)

    #
    # Prepare X & Y edu arrays
    def make_edu_data(self, list_number):

        print("---  Make EDU Data -----")

        X_array = np.empty([0,self.__batch_size,4])
        Y_array = np.empty([0,2])

        for __ticker in self.get_tickers(list_number):

            filename = self.__fileDir + '/data/rawdata/train_' + __ticker + '.csv'

            raw_data = []

            with open(filename, newline='') as f:
                next(f)
                rows = csv.reader(f, delimiter=';', quotechar='|')
                for row in rows:
                    raw_data.append(row)

                X_array_ticker, Y_array_ticker = self.__calculate_col_values(self.__batch_size, raw_data)
            f.close()

            Y_array = np.concatenate((Y_array, Y_array_ticker), axis=0)
            X_array = np.concatenate((X_array, X_array_ticker), axis=0)
            print("->> "+__ticker)

        self.__save_numpy_array("Y_edu", Y_array)
        self.__save_numpy_array("X_edu", X_array)

        print(self.__read_numpy_array('Y_edu').shape)
        print(self.__read_numpy_array('X_edu').shape)


    #
    # Save arrat to file
    def __save_numpy_array(self, name, data):

        index = str(datetime.datetime.now().timestamp()).replace('.','_')
        filename = self.__fileDir + '/data/' + name + '_' + str(index)+'.npy'
        if os.path.isfile(filename):
            os.remove(filename)

        with open(filename, 'wb') as f:
            np.save(f, data)
        f.close()

    #
    # Read array from file
    def __read_numpy_array(self, name):

         filename = self.__fileDir + '/data/' + name + '.npy'
         with open(filename, 'rb') as f:
             return np.load(f)

    #
    # Prepare data from stock rates
    def prepare_data(self, list_number):

        __ticker = 'Unknown file'
        dirOutput = '/data/stocks/'

        #
        #__tickers_array = ['AAL']
        try:

            #for __ticker in self.__get_tickers(dirOutput):
            for __ticker in self.get_tickers(list_number):

                raw_data = []
                with open(self.__fileDir + dirOutput + 'train_' + __ticker + '.csv', newline='') as f:

                    next(f)
                    rows = csv.reader(f, delimiter=',', quotechar='|')
                    row = next(rows)
                    print("---- " + __ticker + " ----------")

                    while True:
                        try:

                            next_row = next(rows)

                            # Find carrier as a change of open in percentages
                            carrier = self.__change_percent(row[3], next_row[3])
                            row = next_row

                            # Find day of year
                            day_of_year = self.__day_of_year(row[0])
                            row.append(day_of_year)
                            row.append(carrier)

                            low_ = self.__change_percent(row[3], row[2])
                            row.append(low_)

                            high_ = self.__change_percent(row[3], row[1])
                            row.append(high_)

                            close_current_ = self.__change_percent(row[3], row[4])
                            row.append(close_current_)

                            raw_data.append(row)

                        except StopIteration:
                            break

                f.close()

                self.__append_to_file(__ticker, raw_data)

        except FileNotFoundError:

            print('Error: File "' + __ticker + '.csv" not found')

    # Check dictionary
    def check_dictionary(self, dirname):

        __ticker = 'Unknown file'
        __d = {}
        count = 0

        print("-------- Dictionary check  ---->> " )
        try:
            for __ticker in self.__get_tickers(dirname):

                raw_data = []
                with open(self.__fileDir + dirname + 'train_' + __ticker + '.csv', newline='') as f:

                    print("---->> " + str(__ticker))

                    next(f)
                    rows = csv.reader(f, delimiter=';', quotechar='|')

                    for row in rows:
                        raw_data.append(row)

                    X_array, Y_array = self.__calculate_col_values(3, raw_data)

                    count += X_array.shape[0]
                    for i in range (0, X_array.shape[0]):

                        key = ";".join(np.array(X_array[i]).flatten().astype(str))

                        if key in __d:
                            __d[key] = __d[key] + 1
                        else:
                            __d[key] = 1
                    print( "Current count :"+str(count))

                f.close()

            print("Common rows : " + str(count) + " dictionary size : " + str(len(__d)))

            i = 0
            if (len(__d) < count):

                for key, value in sorted(__d.items(), key=lambda x: x[1]):

                    if (value > 1):
                        i +=1
                        print("Key :" + str(key) + ", value :" + str(value))

            print("There is dubled " + str(i))

        except FileNotFoundError:
            print('Error: File "' + __ticker + '.csv" not found')


    # Write data to output csv file
    def __append_to_file(self, name, data):

        filename = self.__fileDir + '/data/rawdata/train_' + name + '.csv'
        if os.path.isfile(filename):
            os.remove(filename)

        with open(filename, 'a', newline = '') as csv_out_file:
            output = csv.writer(csv_out_file, delimiter=';')

            output.writerow(['Date', 'Open', 'Low', 'High', 'Close', 'Adj' 'Close', 'Volume',
                            'DAY', 'Carrier', "Low'", "High'", "Close/Current'"])

            for line in data:
                output.writerow(line)

        csv_out_file.close()


    def __change_percent(self, current, next):

        return float(D((float(next) - float(current))/float(current)*100).quantize(D(self.__accuracy), rounding=ROUND_DOWN))

    def __calculate_col_values(self, range_size, raw_data):

        n_array = (np.delete(np.array(raw_data), (0, 5, 6, 7), axis=1)).astype(np.float64)
        data_len = n_array.shape[0]

        Y_array = []
        X_array = []

        for i in range(0, data_len - 2 * range_size + 1):
            end = i + range_size

            # Find max & min in feature slice period
            f_max = np.max(n_array[i + range_size:end + range_size], axis=0)[0]
            f_min = np.min(n_array[i + range_size:end + range_size], axis=0)[1]

            # Find Low & High change in feature slice period
            f_ch_percent_low = self.__change_percent(str(n_array[i:end][-1][3]), f_min)
            f_ch_percent_high = self.__change_percent(str(n_array[i:end][-1][3]), f_max)

            # Remove abs values from array
            X_row = np.delete(n_array[i:end], np.s_[0, 1, 2, 3], 1)
            Y_row = np.array([f_ch_percent_low,f_ch_percent_high])

            Y_array.append(Y_row)
            X_array.append(X_row)

        return np.array(X_array), np.array(Y_array)


    # Convert date to day of year
    def __day_of_year(self, date_str):
        return datetime.strptime(date_str, '%Y-%m-%d').date().timetuple().tm_yday
