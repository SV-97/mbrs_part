///
/// # Example
/// A call might loook like
/// ```sh
/// mbrs_part "part_nr, start, end, size, fs"
/// mbrs_part "1 | 1024B - * | 1GiB | Ext4 bootable"
/// mbrs_part -i in.img -o out.img --expand --drive_signature 0x090b3d33 --table {part_nr =  1, start = "123B", size = "10GiB", fs = "btrfs", bootable = true }
/// ```
use std::{cmp::Ordering, io::Write, str::FromStr};

use clap::{Parser, Subcommand};
use clio::{Input, Output};
use mbrs::{AddrScheme, IncompletePartInfo, Mbr, MbrError, MbrPartTable, PartInfo, PartType};
use num_traits::Num;
use toml::{Table, Value};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PartInfoArg {
    part_nr: Option<usize>,
    pub start_lba: Option<u32>,
    pub end_lba: Option<u32>,
    pub size_lba: u32,
    pub bootable: bool,
    pub part_type: PartType,
}

impl From<PartInfoArg> for IncompletePartInfo {
    fn from(value: PartInfoArg) -> Self {
        Self {
            start_lba: value.start_lba,
            end_lba: value.end_lba,
            size_lba: Some(value.size_lba),
            bootable: value.bootable,
            part_type: value.part_type,
        }
    }
}

trait RoundingIntegerDivide {
    fn int_div(dividend: u128, divisor: u128) -> u128;
}

struct Up;
struct Down;

impl RoundingIntegerDivide for Up {
    fn int_div(dividend: u128, divisor: u128) -> u128 {
        dividend / divisor
    }
}

impl RoundingIntegerDivide for Down {
    fn int_div(dividend: u128, divisor: u128) -> u128 {
        dividend / divisor + if dividend % divisor != 0 { 1 } else { 0 }
    }
}

/// Convert strings like "123512B" into the corresponding number of 512 Byte sectors
/// Valid suffixes:
///     b => bits
///     B => Bytes
///     MiB => Mebibytes
///     GiB => Gibibytes
///     s => sectors
fn data_unit_to_sectors<RoundMode: RoundingIntegerDivide>(s: &str) -> Result<u32, String> {
    let err_msg = |x| move |_| format!("Failed to parse integer {}", x);
    fn parse_int<T: Num + FromStr<Err = <T as Num>::FromStrRadixErr>>(
        s: &str,
    ) -> Result<T, <T as Num>::FromStrRadixErr> {
        if let Some(x) = s.strip_prefix("0x") {
            T::from_str_radix(x, 16)
        } else {
            s.parse()
        }
    }
    // Attempt to find the end of the integer literal
    if let Some(x) = s.strip_suffix("s") {
        parse_int(x).map_err(err_msg(x))
    } else if let Some(x) = s.strip_suffix("MiB") {
        parse_int(x)
            .map_err(err_msg(x))
            .and_then(|mi_bytes: u64| u32::try_from(mi_bytes * 2048).map_err(|e| e.to_string()))
    } else if let Some(x) = s.strip_suffix("GiB") {
        parse_int(x).map_err(err_msg(x)).and_then(|mi_bytes: u64| {
            u32::try_from(mi_bytes * 2048 * 1024).map_err(|e| e.to_string())
        })
    } else if let Some(x) = s.strip_suffix("b") {
        parse_int(x).map_err(err_msg(x)).and_then(|bytes: u128| {
            u32::try_from(RoundMode::int_div(bytes, 512 * 8)).map_err(|e| e.to_string())
        })
    } else if let Some(x) = s.strip_suffix("B") {
        parse_int(x).map_err(err_msg(x)).and_then(|bytes: u128| {
            u32::try_from(RoundMode::int_div(bytes, 512)).map_err(|e| e.to_string())
        })
    } else {
        parse_int(s).map_err(|e| format!("Failed to parse integer `{}`: {}", s, e))
    }
}

impl FromStr for PartInfoArg {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        fn try_parse_with_units<RoundMode: RoundingIntegerDivide>(
            val: Option<&Value>,
        ) -> Result<Option<u32>, String> {
            {
                match val
                    .and_then(Value::as_str)
                    .map(data_unit_to_sectors::<Down>)
                {
                    None => val
                        .and_then(Value::as_integer)
                        .map(|val| u32::try_from(val).map_err(|e| e.to_string())),
                    x => x,
                }
            }
            .transpose()
        }

        let m = Table::from_str(&format!("table = {s}")).map_err(|e| e.to_string())?;

        let m = m.get("table").unwrap();
        let start_lba = try_parse_with_units::<Down>(m.get("start"))?;
        let end_lba = try_parse_with_units::<Up>(m.get("end"))?;
        let size_lba = try_parse_with_units::<Up>(m.get("size"))?;
        // Verify that we have enough information to place the partition into the table
        let size_lba = match (start_lba, end_lba, size_lba) {
            (_, None, None) | (None, _, None) => Err("Partition is undertermined".to_string()),
            (_, _, Some(size)) => Ok(size),
            (Some(start), Some(end), _) if start <= end => Ok(1 + end - start),
            (Some(_start), Some(_end), _) => Err(MbrError::BrokenPartitionBounds.to_string()),
        }?;

        let bootable = m.get("bootable").map(|val| val.as_bool().ok_or("Invalid type for `bootable` in partition table: the partition type has to be given by a string")).transpose()?.unwrap_or(false);
        let part_type_spec = m.get("part_type").or_else(|| m.get("fs")).ok_or("Missing key `part_type` in partition table: you have to specify a partition type")?.as_str().ok_or("Invalid type for `part_type` in partition table: the partition type has to be given by a string")?.split(' ').collect::<Vec<_>>();
        let part_type = match part_type_spec.as_slice() {
            ["btrfs"] | ["ext4"] | ["ext3"] | ["linux"] => Ok(PartType::LinuxNative),
            ["empty"] => Ok(PartType::Empty),
            v @ ["fat12", ..] => Ok(PartType::Fat12 {
                visible: !v.contains(&"invisible"),
            }),
            ["oem"] => Ok(PartType::Oem),
            v @ ["fat16", ..] => Ok(PartType::Fat16 {
                visible: !v.contains(&"invisible"),
                leq32mib: v.contains(&"leq32mib"),
                scheme: if v.contains(&"chs") {
                    AddrScheme::Chs
                } else {
                    AddrScheme::Lba
                },
            }),
            v @ ["extended", ..] => Ok(PartType::Extended {
                scheme: if v.contains(&"chs") {
                    AddrScheme::Chs
                } else {
                    AddrScheme::Lba
                },
            }),
            v @ ["ntfs", ..] | v @ ["hpfs", ..] | v @ ["exfat", ..] => Ok(PartType::ExFAT {
                visible: !v.contains(&"invisible"),
            }),
            v @ ["fat32", ..] => Ok(PartType::Fat32 {
                visible: !v.contains(&"invisible"),
                scheme: if v.contains(&"chs") {
                    AddrScheme::Chs
                } else {
                    AddrScheme::Lba
                },
            }),
            ["win_re"] => Ok(PartType::WindowsRe),
            ["dynamic_disk"] => Ok(PartType::DynamicDisk),
            ["gpfs"] => Ok(PartType::Gpfs),
            ["linux_swap"] | ["swap"] => Ok(PartType::LinuxSwap),
            ["linux_native"] => Ok(PartType::LinuxNative),
            ["intel_rapid_start"] => Ok(PartType::IntelRapidStart),
            ["linux_lvm"] | ["lvm"] => Ok(PartType::LinuxLvm),
            ["free_bsd"] => Ok(PartType::FreeBsd),
            ["open_bsd"] => Ok(PartType::OpenBsd),
            ["net_bsd"] => Ok(PartType::NetBsd),
            ["macos"] => Ok(PartType::MacOs),
            ["solaris"] => Ok(PartType::Solaris),
            ["be_os"] => Ok(PartType::BeOs),
            ["protective_mbr"] => Ok(PartType::ProtectiveMbr),
            ["efi"] => Ok(PartType::Efi),
            ["linux_raid"] => Ok(PartType::LinuxRaid),
            v => Err(format!(
                "Unknown partition type / specifier {}",
                v.join(" ")
            )),
        }?;

        Ok(PartInfoArg {
            start_lba,
            end_lba,
            size_lba,
            bootable,
            part_type,
            part_nr: m
                .get("part_nr")
                .map(|val| {
                    if let Some(i) = val.as_integer() {
                        usize::try_from(i).map_err(|e| e.to_string()).and_then(|i| {
                            if i <= 3 {
                                Ok(i)
                            } else {
                                Err(format!(
                                    "The partition number has to be in the range 0..4 but is {}.",
                                    i
                                ))
                            }
                        })
                    } else {
                        Err(
                            "Invalid type for key `part_nr`. Expected an unsigned integer."
                                .to_string(),
                        )
                    }
                })
                .transpose()?,
        })
    }
}

static TBL_EXPL: &str = r#"
# Omitting fields

Some of the fields may be omitted:
    * `part_nr` - if you omit an explicit partition number the partition will be placed at the first possible position (keeping in mind the size of the partition)
    * `start`, `size`, `end` - if you specify two out of these three, the last one will be calculated to fit. Valid units are `b`, `B`, `MiB`, `GiB`, `s` (sectors).
        Internally the MBR has to use LBA so we try to convert your unit into LBA as best as possible, enlarging the partition by one sector as necessary.
        If you specify only `size` it will append a new partition of that size. 
    * `bootable` - defaults to false

# Filesystems

Valid fileystems are (multiple strings on one line indicate aliases)

    * "btrfs" "ext4" "ext3" "linux"
    * "empty"
    * "fat12"
    * "oem"
    * "fat16"
    * "extended"
    * "ntfs" "hpfs" "exfat"
    * "win_re"
    * "dynamic_disk"
    * "gpfs"
    * "linux_swap" "swap"
    * "linux_native"
    * "intel_rapid_start"
    * "linux_lvm" "lvm"
    * "free_bsd"
    * "open_bsd"
    * "net_bsd"
    * "macos"
    * "solaris"
    * "be_os"
    * "protective_mbr"
    * "efi"
    * "linux_raid"

If the filesystem supports it (this will be verified) you may use the additional specifier

    * "invisible" - mark the partition as invisible
    * "leq32mib" - mark the partition as <= 32MiB (we don't verify this, please don't lie)
    * "chs" - use legacy CHS addressing scheme rather than the default LBA. This limits the number of available addresses a lot.
"#;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Unit {
    GiB,
    MiB,
    KiB,
    Bytes,
    Sectors,
}

impl Unit {
    pub fn val_from_sectors(self, sectors: u32) -> f64 {
        match self {
            Unit::Bytes => (sectors * 512) as f64,
            Unit::Sectors => sectors as f64,
            Unit::KiB => sectors as f64 / 2.,
            Unit::MiB => sectors as f64 / 2048.,
            Unit::GiB => sectors as f64 / (2048. * 1024.),
        }
    }
}

impl FromStr for Unit {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "B" | "Bytes" => Ok(Unit::Bytes),
            "GiB" => Ok(Unit::GiB),
            "MiB" => Ok(Unit::MiB),
            "KiB" => Ok(Unit::KiB),
            "s" | "Sectory" => Ok(Unit::Sectors),
            x => Err(format!("Unknown unit: {}", x)),
        }
    }
}

impl ToString for Unit {
    fn to_string(&self) -> String {
        match self {
            Unit::Bytes => "B",
            Unit::Sectors => "s",
            Unit::KiB => "KiB",
            Unit::MiB => "MiB",
            Unit::GiB => "GiB",
        }
        .to_string()
    }
}

/// Reading, writing and modifying MBRS partition tables
#[derive(Parser, Debug)]
#[command(author, version, about, color = concolor_clap::color_choice())]
struct Cli {
    /// Input image file, use '-' for stdin
    #[arg(long="in", short, default_value = None)]
    input: Option<Input>,

    /// Output file '-' for stdout
    #[arg(long = "out", short, value_parser, default_value = "-")]
    output: Output,

    /// Whether to expand the output to the size of the partitions in the partition table
    #[arg(short, long, default_value_t = false)]
    expand: bool,

    /// Prints a bunch of messages along the way
    #[arg(short, long, default_value_t = false)]
    verbose: bool,

    /// Prints the partition table in the given unit (Bytes, GiB, MiB or Sectors)
    #[arg(short, long, default_value = None)]
    print_table: Option<Unit>,

    /// Drive signature for the MBR (as little endian integer). This value is used as PARTUUID basis by linux.
    /// When omitted this is randomized for new images and not modified for existing images.  
    #[arg(short, long, default_value = None)]
    drive_signature: Option<u32>,

    /// one or multiple TOML dictionaries each specifying a single partition. A fully specified partition may look like:
    ///
    ///     `{part_nr = 1, start = "0xFFB", size = "10_240MiB", end = "10GiB", fs = "btrfs", bootable = true }`
    ///
    /// For more details please see the `show-table-format` subcommand.
    #[arg(short, long, num_args = 0..4, value_delimiter = ';')]
    table: Vec<PartInfoArg>,

    #[command(subcommand)]
    subcommands: Option<Commands>,
}

#[derive(Subcommand, Debug, PartialEq, Eq)]
enum Commands {
    /// Lists detailed format information and supported options of partition tables
    ShowTableFormat,
}

// Returns an index into the given partition table
// fn find_space_for_partition(mbr: &MbrPartTable, part_size: u32) -> Result<(), > {}

/// Returns whether the given partition is consistent with the table (so it doesn't overlap existing partitions or something like that)
fn is_consistent_with_table(partition: &PartInfo, tbl: &MbrPartTable) -> Result<(), String> {
    for j in 0..4 {
        if let Some(previous) = tbl.entries[j] {
            if (previous.start_sector_lba() <= partition.start_sector_lba()
                && partition.start_sector_lba() <= previous.end_sector_lba())
                || (previous.start_sector_lba() <= partition.end_sector_lba()
                    && partition.end_sector_lba() <= previous.end_sector_lba())
            {
                return Err(format!(
                    "Encountered overlapping partitions: {:?} overlaps with {:?}",
                    partition, previous,
                ));
            }
        }
    }
    Ok(())
}

fn by_pairs<T, S, F>(it: impl IntoIterator<Item = T>, f: F) -> impl Iterator<Item = S>
where
    F: Fn(&T, &T) -> S,
{
    let mut it = it.into_iter();
    let first = it.next();
    it.scan(first, move |acc, next| match acc {
        None => None,
        Some(last_val) => {
            let v = f(&last_val, &next);
            *acc = Some(next);
            Some(v)
        }
    })
}

fn main() -> Result<(), String> {
    // let value = r#"x = { part_nr = 1, start="123s", size = "10240MiB", end = "10", part_type = "btrfs", bootable = true}"#.parse::<Table>();
    // match value {
    //     Ok(x) => println!("Success: {}", x),
    //     Err(e) => println!("Err: {}", e),
    // }

    let mut cli = Cli::parse();

    match &cli.subcommands {
        Some(Commands::ShowTableFormat) => {
            println!("{}", TBL_EXPL);
            return Ok(());
        }
        None => (),
    };

    // parse input mbr or create empty MBR
    let mut mbr = match &mut cli.input {
        None => Mbr::default(),
        Some(i) => Mbr::try_from_reader(i.lock()).map_err(|e| e.to_string())?,
    };

    // set drive signature as specified
    let modified_signature = if let Some(ds) = cli.drive_signature {
        mbr.drive_signature = ds.to_le_bytes();
        true
    } else {
        if cli.input.is_none() {
            mbr.drive_signature = rand::random();
            true
        } else {
            false
        }
    };
    if cli.verbose && modified_signature {
        println!(
            "Set drive signature to {:#X}.",
            u32::from_be_bytes(mbr.drive_signature)
        );
    }

    // we wanna start with the first entry (if possible) and have all the optionals at the back
    cli.table.sort_by(|p1, p2| match (p1.part_nr, p2.part_nr) {
        (Some(x), Some(y)) => x.cmp(&y),
        (None, Some(_)) => Ordering::Greater,
        (Some(_), None) => Ordering::Less,
        (None, None) => Ordering::Equal,
    });

    for new_entry in &cli.table {
        if let Some(i) = new_entry.part_nr {
            if mbr.partition_table.entries[i].is_some() && cli.verbose {
                println!("Removing old partition {i} from table in preparation for new partition");
            }
            // remove the old partition from the table so we don't accidentally consider it during our
            // consistency checks etc.
            mbr.partition_table.entries[i] = None;
        }
    }

    // write all the entries to the table
    for new_entry in &cli.table {
        // determine where in the table we'll place this entry
        let i = new_entry.part_nr.map(Ok).unwrap_or_else(|| {
            // find the first empty position
            for j in 0..4 {
                if mbr.partition_table.entries[j].is_none() {
                    return Ok(j);
                }
            }
            // this is unreachable if there's no other bugs
            return Err("Tried to add too many entries to the table: an MBR partition table only supports up to 4 partitions");
        })?;

        // try and see if the partition is fully defined
        if let Ok(p) = PartInfo::try_from(IncompletePartInfo::from(*new_entry)) {
            // if it is, see if it interferes with other entries in the table
            is_consistent_with_table(&p, &mbr.partition_table)?;
            // if it doesn't, write it out
            if cli.verbose {
                if let Some(current) = mbr.partition_table.entries[i] {
                    println!(
                        "Overwriting partition number {} from {:?} to {:?}.",
                        i, current, p
                    );
                } else {
                    println!("Writing partition {:?} to entry {}.", p, i);
                }
            }
            mbr.partition_table.entries[i] = Some(p);
        } else {
            let mut new_entry = *new_entry;
            // we know the size
            let entry_size = new_entry.size_lba;
            // determine the entries already in the table
            let mut current_entries: Vec<_> = mbr
                .partition_table
                .entries
                .iter()
                .enumerate()
                .map(|(i, o)| o.map(|x| (i, x)))
                .flatten()
                .collect();
            // sort them according to their memory layout
            current_entries.sort_by_key(|(_i, p)| p.start_sector_lba());
            // determine free space in-between partitions
            let start = by_pairs(current_entries, |(i, left_part), (_, right_part)| {
                (
                    *i,
                    right_part.start_sector_lba() - left_part.end_sector_lba() - 1,
                )
            })
            .filter(|(_i, space_size)| *space_size >= entry_size)
            .next()
            // if there is a sufficiently large space we place the partition into that space
            .map(|(i, _)| mbr.partition_table.entries[i].unwrap().end_sector_lba() + 1)
            .unwrap_or_else(|| {
                // if there isn't we try placing the new partition after the last partition
                mbr.partition_table
                    .entries
                    .iter()
                    .flatten()
                    .map(|x| x.end_sector_lba() + 1)
                    .max()
                    // if there is no last partition we place the partition after the MBR itself
                    .unwrap_or(1)
            });
            new_entry.start_lba = Some(start);
            // At this point we know that the start is not None
            if cli.verbose {
                println!(
                    "Determined that partition {} with spec {:?} should start at address {:?}.",
                    i,
                    new_entry,
                    new_entry.start_lba.unwrap()
                );
            }

            // the new entry is now guaranteed to be fully defined and we can insert it
            let p = PartInfo::try_from(IncompletePartInfo::from(new_entry))
                .map_err(|e| e.to_string())?;
            // but first we verify that it really fits into the table
            is_consistent_with_table(&p, &mbr.partition_table)?;
            mbr.partition_table.entries[i] = Some(p);
        }
    }

    // Write MBR to output
    cli.output
        .write_all(&<[u8; 512]>::try_from(&mbr).map_err(|e| format!("{:?}", e))?)
        .map_err(|e| e.to_string())?;

    // Print human readable table
    if let Some(unit) = cli.print_table {
        let unit_str = unit.to_string();
        let nr = "MBR Nr.";
        let start = format!("Start [{unit_str}]");
        let stop = format!("Stop [{unit_str}]");
        let size = format!("Size [{unit_str}]");
        let fs = "Filesystem";
        println!("\n{nr:^12} | {start:^12} | {stop:^12} | {size:^12} | {fs:^12}");
        for (i, entry) in mbr.partition_table.entries.iter().enumerate() {
            if let Some(e) = entry {
                println!(
                    "{:12} | {:12.2} | {:12.2} | {:12.2} | {:12?}",
                    i,
                    unit.val_from_sectors(e.start_sector_lba()),
                    unit.val_from_sectors(e.end_sector_lba()),
                    unit.val_from_sectors(e.sector_count_lba()),
                    e.part_type()
                );
            }
        }
    }

    // Expand the output to the necessary size
    if cli.expand {
        let last_address = mbr
            .partition_table
            .entries
            .iter()
            .flat_map(|entry| entry.map(|e| e.end_sector_lba()))
            .sum();

        let unit = cli.print_table.unwrap_or(Unit::GiB);
        if cli.verbose {
            println!(
                "Expanding output to {:.2} {}",
                unit.val_from_sectors(last_address),
                unit.to_string()
            );
        }

        let block = [0; 512];
        for _addresses in 1..last_address {
            cli.output.write_all(&block).map_err(|e| e.to_string())?;
        }
    }
    cli.output.flush().unwrap();

    Ok(())
}

/*
cargo run -- --table \
    '{ start = "0xFFB", size = "10_240MiB", end = "10GiB", fs = "btrfs", bootable = true }' \
    '{ part_nr = 2, start="123s", size = "10240MiB", end = "10", part_type = "ext4", bootable = true}' \
    '{ start="20GiB", size = "10GiB", part_type = "ntfs"}'

cargo run -- --table \
    '{ start = "10B", size = "240MiB", fs = "btrfs", bootable = true }' \
    '{ part_nr = 2, start="500MiB", size = "1GiB", part_type = "ext4", bootable = true}' \
    '{ size = "10GiB", part_type = "ntfs"}'

cargo run -- -e -v -p GiB --out out.img --table \
    '{ start = "10B", size = "240MiB", fs = "btrfs", bootable = true }' \
    '{ part_nr = 2, start="500MiB", size = "1GiB", part_type = "ext4", bootable = true}' \
    '{ size = "10GiB", part_type = "ntfs"}'
*/
