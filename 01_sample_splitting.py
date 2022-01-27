import shutil
import subprocess
import os
import pathlib
from datetime import datetime
import gzip


def split_samples(acq_id, barcode_file, raw_dir, output_dir, verbose=1, n_mismatch=1,
                  r1_part=None, r2_part=None):
    """Split raw fastq data according to sample barcodes

    This unzips raw fastq.gz files, cuts the two reads if needed (using `r1_part` and
    `r2_part`) and runs fastx_barcode_splitter.

    The output directory will contain a .txt file per barcode and a file named
    `barcode_splitter_log.txt` containing the summary output from the fastx function.

    Args:
        acq_id (str): Acquisition ID. Only file starting with this id will be unzipped
        barcode_file (str or Path): path to the file containing the list of barcodes
        raw_dir: (str or Path): Path to the folder containing the fastq.gz files. It
            should contain two files starting with acq_id, one for read 1 with `R1` in
            its name and one for read 2, with `R2` in its name
        output_dir (str or Path): [optional] Directory to save the output, if None,
            will save in the current working directory
        n_mismatch (int): [optional] number of mismatches accepted. Default to 1
        r1_part (None or (int, int)): [optional] part of the read 1 sequence to keep,
            None to keep the full read, [beginning, end] otherwise
        r2_part (int, int): [optional] same as r1_part but for read 2
        verbose (int): Level of feedback printed. 0 for nothing, 1 for steps,
            2 for steps and full output


    Returns:
        None
    """
    if verbose:
        tstart = datetime.now()
    fastq_files = unzip_fastq(raw_dir, acq_id=acq_id, target_dir=output_dir,
                              overwrite=False)
    # make sure we have read1 and read2
    read_files = dict()
    for read_number in [1, 2]:
        good_file = [r for r in fastq_files.keys() if 'R%d' % read_number in r]
        if not good_file:
            raise IOError('Could not find read %d file' % read_number)
        elif len(good_file) > 1:
            raise IOError('Found multiple files for read %d:\n%s' % (read_number,
                                                                     good_file))
        else:
            read_files[read_number] = fastq_files[good_file[0]]
    run_bc_splitter(read1_file=read_files[1], read2_file=read_files[2],
                    barcode_file=barcode_file, n_mismatch=n_mismatch, r1_part=r1_part,
                    r2_part=r2_part, output_dir=output_dir, verbose=verbose)
    # remove fastq files:
    for f in fastq_files.values():
        os.remove(f)

    if verbose:
        tend = datetime.now()
        print('That took %s' % (tend - tstart), flush=True)


def unzip_fastq(source_dir, acq_id, target_dir=None, overwrite=False, verbose=1):
    """Unzip fastq.gz files

    Args:
        source_dir (str or pathlib.Path): Path to the folder containing the
            fastq.gz files.
        acq_id (str): Acquisition ID. Only file starting with this id will be unzipped
        target_dir (str or pathlib.Path): Path to directory to write the output. If
            None (default), will write in source_dir
        overwrite (bool): Overwrite target if it already exists. Default to False
        verbose (int): 0 do not print anything. 1 print progress

    Returns:
        out_files (dict): Dictionary of output files with file name as keys and full
            path as values
    """
    source_dir = pathlib.Path(source_dir)
    assert source_dir.is_dir()

    if target_dir is None:
        target_dir = source_dir
    else:
        target_dir = pathlib.Path(target_dir)
        if not target_dir.is_dir():
            target_dir.mkdir(mode=774)
    out_files = dict()
    for gz_file in source_dir.glob('{0}*.fastq.gz'.format(acq_id)):
        target_file = target_dir / gz_file.stem
        if target_file.exists() and (not overwrite):
            raise IOError('%s already exists. Use overwrite to replace' % target_file)
        if verbose:
            t = datetime.now()
            print('Unzipping %s (%s)' % (gz_file, t.strftime('%H:%M:%S')))
        with gzip.open(gz_file, 'rb') as source, \
            open(target_file, 'wb') as target:
            shutil.copyfileobj(source, target)
        out_files[target_file.stem] = target_file
    return out_files


def run_bc_splitter(read1_file, read2_file, barcode_file, n_mismatch=1, r1_part=None,
                  r2_part=None, output_dir=None, verbose=1):
    """Split samples using Barcode splitter

    Format data and calls barcode splitter. It will also save the summary output of
    barcode splitter in 'barcode_splitter_log.txt' in the same directory as the other
    output file.

    Args:
        read1_file (str or Path): path to the fastq file of the first read
        read2_file (str or Path): path to the fastq file of the second read
        barcode_file (str or Path): path to the file containing the list of barcodes
        n_mismatch (int): [optional] number of mismatches accepted. Default to 1
        r1_part (None or (int, int)): [optional] part of the read 1 sequence to keep,
            None to keep the full read, [beginning, end] otherwise
        r2_part (int, int): [optional] same as r1_part but for read 2
        output_dir (str or Path): [optional] Directory to save the output, if None,
            will save in the current working directory
        verbose (int): Level of feedback printed. 0 for nothing, 1 for steps,
            2 for steps and full output

    Returns:
        None
    """
    if verbose:
        t = datetime.now()
        print('Split sequence and merge reads (%s)' % t.strftime('%H:%M:%S'), flush=True)
    if output_dir is None:
        output_dir = os.getcwd()
    output_dir = pathlib.Path(output_dir)
    if not output_dir.is_dir():
        output_dir.mkdir(mode=774)

    # barcode splitter expects a file with the sequence number on one line and the
    # sequence on the next. We will do that
    temp_file = output_dir / 'barcode_splitter_input.fasta'
    with open(read1_file, 'r') as read1, \
        open(read2_file, 'r') as read2, \
        open(temp_file, 'w') as target:
        # read on line out of 4
        n_reads = 1
        for il, (r1, r2) in enumerate(zip(read1, read2)):
            if il % 4 == 1:
                # crop sequence if needed
                if r1_part is not None:
                    r1 = r1[r1_part[0]: r1_part[1]]
                if r2_part is not None:
                    r2 = r2[r2_part[0]: r2_part[1]]
                # concatenate and write to temporary file
                full_read = r1.strip() + r2.strip() + '\n'
                target.write('> {0}\n{1}'.format(n_reads, full_read))
                n_reads += 1


    #split dataset according to inline indexes using fastx toolkit; this by default allows up to 1 missmatch. we could go higher if we want, though maybe not neccessary
    # now run barcode splitter on that
    if verbose:
        t = datetime.now()
        print('Barcode splitter (%s)' % t.strftime('%H:%M:%S'), flush=True)
    with open(temp_file, 'r') as file_input:
        out = subprocess.run(['fastx_barcode_splitter.pl', '--bcfile', str(barcode_file),
                              '--prefix', str(output_dir) + os.path.sep, '--eol',
                              '--suffix', '.txt', '--mismatches', str(n_mismatch)],
                             stdin=file_input, capture_output=True)
    if out.stderr:
        raise IOError('Barcode splitter raised an error:\n{0}', out.stderr)

    log_file = output_dir / 'barcode_splitter_log.txt'
    with open(log_file, 'wb') as log:
        log.write(out.stdout)

    if verbose > 1:
        print(out.stdout.decode(), flush=True)
    if verbose:
        t = datetime.now()
        print('Sample splitting done, removing temporary file (%s)' %
              t.strftime('%H:%M:%S'))
    os.remove(temp_file)





if __name__ == '__main__':
    verbose = 1  # 0 is silent, 1 prints progress, >2 also prints barcode splitter output
    # Script to take raw data files and split by sample

    camp_root = pathlib.Path('/camp/lab/znamenskiyp/home/shared/projects/turnerb_MAPseq')
    raw = camp_root / 'Sequencing/Raw_data/BRAC5676.1h/trial/fastq'
    out_dir = camp_root / 'Sequencing/Processed_data/BRAC5676.1h/trial/temp_test'
    acqid = 'TUR4405A1'
    barcodes = camp_root / 'Sequencing/Reference_files/sample_barcodes.txt'

    split_samples(raw_dir=raw, output_dir=out_dir, acq_id=acqid, barcode_file=barcodes,
                  n_mismatch=1, r1_part=None, r2_part=(0,30), verbose=1)

