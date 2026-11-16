//! Phone inventory (ARPAbet, same order as the `tts` crate) and broad classes.

pub const N_PHONES: usize = 41;
pub const SIL: u8 = 0;

pub const PHONE_NAMES: [&str; N_PHONES] = [
    "SIL", "IY", "IH", "EH", "AE", "AA", "AO", "UH", "UW", "AH", "AX", "ER", "EY", "AY", "OY", "AW",
    "OW", "P", "B", "T", "D", "K", "G", "CH", "JH", "F", "V", "TH", "DH", "S", "Z", "SH", "ZH", "HH",
    "M", "N", "NG", "L", "R", "W", "Y",
];

/// Parses an ARPAbet symbol in any case, with or without a stress digit. Silence markers
/// used by label files (`pau`, `ssil`, `h#`) map to [`SIL`].
pub fn phone_index(name: &str) -> Option<u8> {
    let name = name.trim_end_matches(|c: char| c.is_ascii_digit());
    if ["pau", "ssil", "h#", "sil"].iter().any(|s| name.eq_ignore_ascii_case(s)) {
        return Some(SIL);
    }
    PHONE_NAMES.iter().position(|p| p.eq_ignore_ascii_case(name)).map(|i| i as u8)
}

pub fn phone_name(ph: u8) -> &'static str {
    PHONE_NAMES.get(ph as usize).copied().unwrap_or("?")
}

/// Merges the reduced vowel into AH, the usual convention when scoring phone recognition.
pub fn fold(ph: u8) -> u8 {
    if ph == 10 {
        9
    } else {
        ph
    }
}

pub const N_BROAD: usize = 7;
pub const BROAD_NAMES: [&str; N_BROAD] = ["silence", "vowel", "glide", "nasal", "stop", "strong-fric", "weak-fric"];

/// Manner-of-articulation class of a phone.
pub fn broad_class(ph: u8) -> u8 {
    match ph {
        0 => 0,
        1..=16 => 1,
        17..=22 => 4,
        23 | 24 | 29..=32 => 5,
        25..=28 | 33 => 6,
        34..=36 => 3,
        _ => 2,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_symbols_and_classes() {
        assert_eq!(phone_index("AH1"), Some(9));
        assert_eq!(phone_index("ax"), Some(10));
        assert_eq!(phone_index("pau"), Some(SIL));
        assert_eq!(phone_index("zz"), None);
        assert_eq!(fold(phone_index("AX").unwrap()), phone_index("AH").unwrap());
        let class = |p: &str| BROAD_NAMES[broad_class(phone_index(p).unwrap()) as usize];
        assert_eq!(class("S"), "strong-fric");
        assert_eq!(class("CH"), "strong-fric");
        assert_eq!(class("F"), "weak-fric");
        assert_eq!(class("HH"), "weak-fric");
        assert_eq!(class("NG"), "nasal");
        assert_eq!(class("W"), "glide");
        assert_eq!(class("OW"), "vowel");
        assert_eq!(class("G"), "stop");
    }
}
