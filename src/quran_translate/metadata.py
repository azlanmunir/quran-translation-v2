"""Static Quran metadata used for validation and export headings."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SurahInfo:
    number: int
    transliteration: str
    meaning: str
    ayah_count: int

    @property
    def heading(self) -> str:
        return f"{self.number:03d}. {self.transliteration} - {self.meaning}"


SURAHS: tuple[SurahInfo, ...] = (
    SurahInfo(1, "Al-Fatihah", "The Opening", 7),
    SurahInfo(2, "Al-Baqarah", "The Cow", 286),
    SurahInfo(3, "Aal-Imran", "Family of Imran", 200),
    SurahInfo(4, "An-Nisa", "The Women", 176),
    SurahInfo(5, "Al-Ma'idah", "The Table", 120),
    SurahInfo(6, "Al-An'am", "The Cattle", 165),
    SurahInfo(7, "Al-A'raf", "The Heights", 206),
    SurahInfo(8, "Al-Anfal", "The Spoils", 75),
    SurahInfo(9, "At-Tawbah", "Repentance", 129),
    SurahInfo(10, "Yunus", "Jonah", 109),
    SurahInfo(11, "Hud", "Hud", 123),
    SurahInfo(12, "Yusuf", "Joseph", 111),
    SurahInfo(13, "Ar-Ra'd", "Thunder", 43),
    SurahInfo(14, "Ibrahim", "Abraham", 52),
    SurahInfo(15, "Al-Hijr", "The Rocky Tract", 99),
    SurahInfo(16, "An-Nahl", "The Bee", 128),
    SurahInfo(17, "Al-Isra", "Night Journey", 111),
    SurahInfo(18, "Al-Kahf", "The Cave", 110),
    SurahInfo(19, "Maryam", "Mary", 98),
    SurahInfo(20, "Ta-Ha", "Ta-Ha", 135),
    SurahInfo(21, "Al-Anbiya", "The Prophets", 112),
    SurahInfo(22, "Al-Hajj", "The Pilgrimage", 78),
    SurahInfo(23, "Al-Mu'minun", "The Believers", 118),
    SurahInfo(24, "An-Nur", "The Light", 64),
    SurahInfo(25, "Al-Furqan", "The Criterion", 77),
    SurahInfo(26, "Ash-Shu'ara", "The Poets", 227),
    SurahInfo(27, "An-Naml", "The Ant", 93),
    SurahInfo(28, "Al-Qasas", "The Stories", 88),
    SurahInfo(29, "Al-Ankabut", "The Spider", 69),
    SurahInfo(30, "Ar-Rum", "The Romans", 60),
    SurahInfo(31, "Luqman", "Luqman", 34),
    SurahInfo(32, "As-Sajdah", "Prostration", 30),
    SurahInfo(33, "Al-Ahzab", "The Confederates", 73),
    SurahInfo(34, "Saba", "Sheba", 54),
    SurahInfo(35, "Fatir", "Originator", 45),
    SurahInfo(36, "Ya-Sin", "Ya-Sin", 83),
    SurahInfo(37, "As-Saffat", "Those in Ranks", 182),
    SurahInfo(38, "Sad", "Sad", 88),
    SurahInfo(39, "Az-Zumar", "The Troops", 75),
    SurahInfo(40, "Ghafir", "The Forgiver", 85),
    SurahInfo(41, "Fussilat", "Explained in Detail", 54),
    SurahInfo(42, "Ash-Shura", "Consultation", 53),
    SurahInfo(43, "Az-Zukhruf", "Ornaments", 89),
    SurahInfo(44, "Ad-Dukhan", "The Smoke", 59),
    SurahInfo(45, "Al-Jathiyah", "Crouching", 37),
    SurahInfo(46, "Al-Ahqaf", "The Dunes", 35),
    SurahInfo(47, "Muhammad", "Muhammad", 38),
    SurahInfo(48, "Al-Fath", "The Victory", 29),
    SurahInfo(49, "Al-Hujurat", "The Chambers", 18),
    SurahInfo(50, "Qaf", "Qaf", 45),
    SurahInfo(51, "Adh-Dhariyat", "Scattering Winds", 60),
    SurahInfo(52, "At-Tur", "The Mount", 49),
    SurahInfo(53, "An-Najm", "The Star", 62),
    SurahInfo(54, "Al-Qamar", "The Moon", 55),
    SurahInfo(55, "Ar-Rahman", "The Most Merciful", 78),
    SurahInfo(56, "Al-Waqi'ah", "The Inevitable", 96),
    SurahInfo(57, "Al-Hadid", "Iron", 29),
    SurahInfo(58, "Al-Mujadila", "The Pleading", 22),
    SurahInfo(59, "Al-Hashr", "The Exile", 24),
    SurahInfo(60, "Al-Mumtahanah", "She Who is Tested", 13),
    SurahInfo(61, "As-Saff", "The Ranks", 14),
    SurahInfo(62, "Al-Jumu'ah", "Friday", 11),
    SurahInfo(63, "Al-Munafiqun", "The Hypocrites", 11),
    SurahInfo(64, "At-Taghabun", "Mutual Disillusion", 18),
    SurahInfo(65, "At-Talaq", "Divorce", 12),
    SurahInfo(66, "At-Tahrim", "Prohibition", 12),
    SurahInfo(67, "Al-Mulk", "Sovereignty", 30),
    SurahInfo(68, "Al-Qalam", "The Pen", 52),
    SurahInfo(69, "Al-Haqqah", "The Reality", 52),
    SurahInfo(70, "Al-Ma'arij", "The Ascending Ways", 44),
    SurahInfo(71, "Nuh", "Noah", 28),
    SurahInfo(72, "Al-Jinn", "The Jinn", 28),
    SurahInfo(73, "Al-Muzzammil", "The Enshrouded", 20),
    SurahInfo(74, "Al-Muddaththir", "The Cloaked", 56),
    SurahInfo(75, "Al-Qiyamah", "Resurrection", 40),
    SurahInfo(76, "Al-Insan", "Man", 31),
    SurahInfo(77, "Al-Mursalat", "Those Sent Forth", 50),
    SurahInfo(78, "An-Naba", "The Tidings", 40),
    SurahInfo(79, "An-Nazi'at", "Those Who Pull Out", 46),
    SurahInfo(80, "Abasa", "He Frowned", 42),
    SurahInfo(81, "At-Takwir", "The Overthrowing", 29),
    SurahInfo(82, "Al-Infitar", "The Cleaving", 19),
    SurahInfo(83, "Al-Mutaffifin", "Defrauders", 36),
    SurahInfo(84, "Al-Inshiqaq", "The Splitting", 25),
    SurahInfo(85, "Al-Buruj", "The Constellations", 22),
    SurahInfo(86, "At-Tariq", "The Night Comer", 17),
    SurahInfo(87, "Al-A'la", "The Most High", 19),
    SurahInfo(88, "Al-Ghashiyah", "The Overwhelming", 26),
    SurahInfo(89, "Al-Fajr", "The Dawn", 30),
    SurahInfo(90, "Al-Balad", "The City", 20),
    SurahInfo(91, "Ash-Shams", "The Sun", 15),
    SurahInfo(92, "Al-Layl", "The Night", 21),
    SurahInfo(93, "Ad-Duha", "The Morning Hours", 11),
    SurahInfo(94, "Ash-Sharh", "The Relief", 8),
    SurahInfo(95, "At-Tin", "The Fig", 8),
    SurahInfo(96, "Al-Alaq", "The Clot", 19),
    SurahInfo(97, "Al-Qadr", "The Power", 5),
    SurahInfo(98, "Al-Bayyinah", "The Clear Proof", 8),
    SurahInfo(99, "Az-Zalzalah", "The Earthquake", 8),
    SurahInfo(100, "Al-Adiyat", "The Chargers", 11),
    SurahInfo(101, "Al-Qari'ah", "The Calamity", 11),
    SurahInfo(102, "At-Takathur", "Competition", 8),
    SurahInfo(103, "Al-Asr", "Time", 3),
    SurahInfo(104, "Al-Humazah", "The Slanderer", 9),
    SurahInfo(105, "Al-Fil", "The Elephant", 5),
    SurahInfo(106, "Quraysh", "Quraysh", 4),
    SurahInfo(107, "Al-Ma'un", "Small Kindnesses", 7),
    SurahInfo(108, "Al-Kawthar", "Abundance", 3),
    SurahInfo(109, "Al-Kafirun", "The Disbelievers", 6),
    SurahInfo(110, "An-Nasr", "Divine Support", 3),
    SurahInfo(111, "Al-Masad", "The Palm Fiber", 5),
    SurahInfo(112, "Al-Ikhlas", "Sincerity", 4),
    SurahInfo(113, "Al-Falaq", "The Daybreak", 5),
    SurahInfo(114, "An-Nas", "Mankind", 6),
)

SURAH_BY_NUMBER = {surah.number: surah for surah in SURAHS}


def surah_info(number: int) -> SurahInfo:
    return SURAH_BY_NUMBER[number]

